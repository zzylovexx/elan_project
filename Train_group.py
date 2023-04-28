'''
可以跟train_cond整理成一個
'''
from torch_lib.Dataset_extra import *
from torch_lib.Model import Model, OrientationLoss
from library.ron_utils import *

import torch
import torch.nn as nn
from torchvision.models import vgg
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--label-dir", default="/label_2/", help='dir name of the labels')
parser.add_argument("--weights-name", required=True, help='output = weights-name_epoch.pkl') 
parser.add_argument("--group", action='store_true', help='with group label or not')
parser.add_argument("--latest", action='store_true', help='continue training or not')
parser.add_argument("--warm-up", type=int, default=10, help='warm up before adding group loss')
parser.add_argument("--device", type=int, default=0, help='select cuda index')
parser.add_argument("--log-dir", default='log', help='tensorboard log-saved path')

def main():
    
    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_name
    group_training = FLAGS.group
    use_latest=FLAGS.latest
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    log_dir = FLAGS.log_dir

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    # hyper parameters
    epochs = 100
    batch_size = 16 #64 worse than 8
    alpha = 0.6
    w = 0.4

    print("Loading all detected objects in dataset...")
        
    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path, FLAGS.label_dir, theta=True, bins=2)

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features).to(device)
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # milestones:調整lr的epoch數，gamma:decay factor (https://hackmd.io/@Hong-Jia/H1hmbNr1d)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_SGD, milestones=[i for i in range(10, epochs, 10)], gamma=0.5)
    conf_loss_func = nn.CrossEntropyLoss().to(device)
    dim_loss_func = nn.MSELoss().to(device)
    orient_loss_func = OrientationLoss
    if group_training==True:
        print("< Train with GroupLoss >")
        group_loss_func = stdGroupLoss

    # load any previous weights
    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights_group/'
    latest_model = None
    first_epoch = 0
    passes = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass
    
    if use_latest==True:
        if latest_model is not None:
            checkpoint = torch.load(model_path + latest_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
            first_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            passes = checkpoint['passes']
            print('Found previous checkpoint: %s at epoch %s'%(latest_model, first_epoch))
            print('Resuming training....')

    total_num_batches = int(len(dataset) / batch_size)#len(dataset)=40570
    
    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        GT_alpha_list = list()
        pred_alpha_list = list()
        for local_batch, local_labels in generator:

            truth_orient = local_labels['Orientation'].float().to(device)
    
            truth_conf = local_labels['Confidence'].long().to(device)
            truth_dim = local_labels['Dimensions'].float().to(device)

            local_batch=local_batch.float().to(device)
            [orient, conf, dim] = model(local_batch)

            # return list type
            GT_alpha, pred_alpha = get_alpha(orient, truth_orient, truth_conf)
            GT_alpha_list += GT_alpha
            pred_alpha_list += pred_alpha
            
            orient_loss = orient_loss_func(orient, truth_orient, truth_conf).to(device)
            dim_loss = dim_loss_func(dim, truth_dim)
            #added loss
            if group_training==True and epoch > warm_up:
                truth_Theta = local_labels['Theta'].float().to(device)
                truth_Ry = local_labels['Ry'].float().to(device)
                truth_group = local_labels['Group'].float().to(device)
                group_loss = group_loss_func(orient, truth_conf, truth_group, truth_Theta, truth_Ry, device)
                

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)
            
            loss_theta = conf_loss + w * orient_loss #w=0.4
            loss = alpha * dim_loss + loss_theta#alpha=0.6
            
            if group_training==True and epoch > warm_up:
                loss += 0.3 * group_loss
                #loss += 0.2 * orient_loss #0.4+0.3=0.7 (added alpha weight)

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()


            if passes % 200 == 0 and group_training==True and epoch> warm_up:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[orient_loss:%.4f],[dim_loss:%.4f],[conf_loss:%.4f],[group_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(),orient_loss.item(),dim_loss.item(),conf_loss.item(),group_loss.item()))
                
                writer.add_scalar('pass/orient_loss', orient_loss, passes//200)
                writer.add_scalar('pass/dim_loss', dim_loss, passes//200)
                writer.add_scalar('pass/conf_loss', conf_loss, passes//200)
                writer.add_scalar('pass/loss_theta', loss_theta, passes//200) #conf_loss + w*orient_loss
                writer.add_scalar('pass/total_loss', loss, passes//200) #conf_loss + 0.4*orient_loss + 0.6*dim_loss + 0.6*offset_loss
                writer.add_scalar('pass/group_loss', group_loss, passes//200)

            elif passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[orient_loss:%.4f],[dim_loss:%.4f],[conf_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(),orient_loss.item(),dim_loss.item(),conf_loss.item()))
                
                writer.add_scalar('pass/orient_loss', orient_loss, passes//200)
                writer.add_scalar('pass/dim_loss', dim_loss, passes//200)
                writer.add_scalar('pass/conf_loss', conf_loss, passes//200)
                writer.add_scalar('pass/loss_theta', loss_theta, passes//200) #conf_loss + w*orient_loss
                writer.add_scalar('pass/total_loss', loss, passes//200) #conf_loss + 0.4*orient_loss + 0.6*dim_loss + 0.6*offset_loss
                writer.add_scalar('pass/group_loss', 0, passes//200) #conf_loss + 0.4*orient_loss + 0.6*dim_loss + 0.6*offset_loss

            passes += 1
            curr_batch += 1

        cos_delta = angle_criterion(pred_alpha_list, GT_alpha_list)
        print(f'Epoch:{epoch} lr = {scheduler.get_last_lr()[0]}')
        print(f'cos_delta: sum={cos_delta.sum()}, mean:{cos_delta.mean()}') #sum=40570 is best, mean=1 is best
        writer.add_scalar('epoch/cos_delta_sum', cos_delta.sum(), epoch)
        writer.add_scalar('epoch/cos_delta_mean', cos_delta.mean(), epoch)
        #write every epoch
        writer.add_scalar('epoch/orient_loss', orient_loss, epoch)
        writer.add_scalar('epoch/dim_loss', dim_loss, epoch)
        writer.add_scalar('epoch/conf_loss', conf_loss, epoch)
        writer.add_scalar('epoch/loss_theta', loss_theta, epoch) #conf_loss + w*orient_loss
        writer.add_scalar('epoch/total_loss', loss, epoch) #conf_loss + 0.4*orient_loss + 0.6*dim_loss + 0.6*offset_loss
        if epoch > warm_up:
            writer.add_scalar('epoch/group_loss', group_loss, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        #tensorboard --logdir=./{log_foler} --port 8123

        # save after every 10 epochs
        scheduler.step()
        if epoch % 10 == 0:
            name = model_path + f'{weights_path}epoch_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss,
                    'passes': passes
                    }, name)
            print("====================")
            
    writer.close()

if __name__=='__main__':
    main()
