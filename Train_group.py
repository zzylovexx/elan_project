from torch_lib.Dataset_theta import *
from torch_lib.Model import Model, OrientationLoss
from library.ron_utils import RyGroupLoss, stdGroupLoss

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--label-dir", default="/label_2_new_group/", help='dir name of the labels')
parser.add_argument("--group", action='store_true', help='with group label or not')
parser.add_argument("--latest", action='store_true', help='continue training or not')
parser.add_argument("--warm-up", default=10, help='warm up before adding group loss')

def main():

    writer = SummaryWriter('./log_group')
    FLAGS = parser.parse_args()
    
    # hyper parameters
    epochs = 50
    batch_size = 16 #64 worse than 8
    alpha = 0.6
    w = 0.4

    print("Loading all detected objects in dataset...")
        

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path, FLAGS.label_dir, theta=True)

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features).cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss
    if FLAGS.group==True:
        print("< Train with GroupLoss >")
        group_loss_func = stdGroupLoss

    # load any previous weights
    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights_group/'
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
            latest_model = 'w05_epoch_10_before_group.pkl'
        except:
            pass
    use_latest=FLAGS.latest
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
    passes = 0
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        
        for local_batch, local_labels in generator:

            truth_orient = local_labels['Orientation'].float().cuda()
    
            truth_conf = local_labels['Confidence'].long().cuda()
            truth_dim = local_labels['Dimensions'].float().cuda()

            local_batch=local_batch.float().cuda()
            [orient, conf, dim] = model(local_batch)

            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            dim_loss = dim_loss_func(dim, truth_dim)
            #added loss
            if FLAGS.group==True and epoch > warm_up:
                truth_Theta = local_labels['Theta'].float().cuda()
                truth_Ry = local_labels['Ry'].float().cuda()
                truth_group = local_labels['Group'].float().cuda()
                group_loss = group_loss_func(orient, truth_conf, truth_group, truth_Theta, truth_Ry)
                

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            loss_theta = conf_loss + w * orient_loss #w=0.4
            loss = alpha * dim_loss + loss_theta#alpha=0.6
            
            if FLAGS.group==True and epoch > warm_up:
                loss += 0.5 * group_loss 

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()


            if passes % 200 == 0 and FLAGS.group==True and epoch> warm_up:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[orient_loss:%.4f],[dim_loss:%.4f],[conf_loss:%.4f],[group_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(),orient_loss.item(),dim_loss.item(),conf_loss.item(),group_loss.item()))

            elif passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[orient_loss:%.4f],[dim_loss:%.4f],[conf_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(),orient_loss.item(),dim_loss.item(),conf_loss.item()))


            passes += 1
            curr_batch += 1

        #write every epoch
        writer.add_scalar('loss/orient_loss', orient_loss, epoch)
        writer.add_scalar('loss/dim_loss', dim_loss, epoch)
        writer.add_scalar('loss/conf_loss', conf_loss, epoch)
        writer.add_scalar('loss/loss_theta', loss_theta, epoch) #conf_loss + w*orient_loss
        writer.add_scalar('loss/total_loss', loss, epoch) #conf_loss + 0.4*orient_loss + 0.6*dim_loss + 0.6*offset_loss
        if epoch > warm_up:
            writer.add_scalar('group_loss', group_loss, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        #tensorboard --logdir=./{log_foler} --port 8123

        # save after every 10 epochs
        if epoch % 5 == 0:
            name = model_path + 'w05_epoch_%s.pkl' % epoch
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
