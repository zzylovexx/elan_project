from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss, GroupLoss


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data


import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--label-dir", default="/label_2_group/", help='dir name of the labels')
parser.add_argument("--group", default="True", help='with group label or not')
#python Train_group.py --label-dir="/label_2_group/" --group

def main():

    FLAGS = parser.parse_args()
    # hyper parameters
    epochs = 20
    batch_size = 64 #128 overloaded
    alpha = 0.6
    w = 0.4

    print("Loading all detected objects in dataset...")
    if FLAGS.group:
        print("< Train with GroupLoss >")

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path, FLAGS.label_dir)

    params = {'batch_size': batch_size,
              'shuffle': not FLAGS.group,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features).cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss
    if FLAGS.group:
        group_loss_func = GroupLoss

    # load any previous weights
    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights_group/'
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass
    use_latest=False
    if use_latest==True:
        if latest_model is not None:
            checkpoint = torch.load(model_path + latest_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
            first_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print('Found previous checkpoint: %s at epoch %s'%(latest_model, first_epoch))
            print('Resuming training....')



    total_num_batches = int(len(dataset) / batch_size)#len(dataset)=40570

    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:

            truth_orient = local_labels['Orientation'].float().cuda()
    
            truth_conf = local_labels['Confidence'].long().cuda()
            truth_dim = local_labels['Dimensions'].float().cuda()

            local_batch=local_batch.float().cuda()
            [orient, conf, dim] = model(local_batch)

            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            dim_loss = dim_loss_func(dim, truth_dim)
            #added loss
            if FLAGS.group:
                truth_group = local_labels['Group'].float().cuda()
                group_loss = group_loss_func(orient, truth_orient, truth_conf, truth_group)

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            loss_theta = conf_loss + w * orient_loss #w=0.4
            loss = alpha * dim_loss + loss_theta#alpha=0.6

            if FLAGS.group:
                #loss += 0.2 * group_loss # cal_orient, rotation_y (sin_loss before 228)
                loss += 0.5 * group_loss # cal_orient sin_loss in 228
                #loss += group_loss # std_loss
                #loss += 0.5*group_loss # cos_loss

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()


            if passes % 200 == 0 and FLAGS.group:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[orient_loss:%.4f],[dim_loss:%.4f],[conf_loss:%.4f],[group_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(),orient_loss.item(),dim_loss.item(),conf_loss.item(),group_loss.item()))
                passes = 0
            elif passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[orient_loss:%.4f],[dim_loss:%.4f],[conf_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(),orient_loss.item(),dim_loss.item(),conf_loss.item()))
                passes = 0

            passes += 1
            curr_batch += 1
        # save after every 10 epochs
        if epoch % 20 == 0:
            name = model_path + 'epoch_%s.pkl' % epoch
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss
                    }, name)
            print("====================")

if __name__=='__main__':
    main()
