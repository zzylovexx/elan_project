from torch_lib.KITTI_Dataset import *
from torch_lib.Model_heading_bin import Model, compute_residual_loss
from library.ron_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=2023, help='keep seeds to represent same result')
# path setting
parser.add_argument("--weights-path", required=True, help='weights_folder/weights_name.pkl, ie.weights/bin4_dim4_group.pkl')
parser.add_argument("--latest-weights", default=None, help='only input the weights-name.pkl') #in the same folder as above
parser.add_argument("--log-dir", default='log', help='tensorboard log-saved path')

#training setting
parser.add_argument("--device", type=int, default=0, help='select cuda index')
parser.add_argument("--epoch", type=int, default=50, help='epoch num')
parser.add_argument("--warm-up", type=int, default=10, help='warm up before adding group loss')

#parser.add_argument("--batch-size", type=int, default=16, help='batch size')

# hyper-parameter (group | bin | cond)
parser.add_argument("--bin", type=int, default=4, help='heading bin num')
parser.add_argument("--group", action='store_true', help='if True, add stdGroupLoss')
parser.add_argument("--cond", action='store_true', help='if True, 4-dim with theta_ray | boxH_2d ')
# TO BE ADDED (Loss的weights比例alpha, w of groupLoss, LRscheduler: milestone, gamma )

def main():
    
    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    

    os.makedirs(FLAGS.log_dir, exist_ok=True)
    writer = SummaryWriter(FLAGS.log_dir)
    epochs = FLAGS.epoch
    batch_size = 16 #64 worse than 8
    alpha = 0.6

    # model
    print("Loading all detected objects in dataset...")
    print('Kitti dataset')
    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset_L = Dataset(train_path, camera='left', condition=FLAGS.cond, num_heading_bin=FLAGS.bin)
    dataset_R = Dataset(train_path, camera='right', condition=FLAGS.cond, num_heading_bin=FLAGS.bin)
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    train_loader_L = data.DataLoader(dataset_L, **params)
    train_loader_R = data.DataLoader(dataset_R, **params)

    my_vgg = vgg.vgg19_bn(weights='DEFAULT')
    if is_cond:
        print("< 4-dim input, Theta_ray as Condition >")
        my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
    model = Model(features=my_vgg.features, bins=bin_num).to(device)
    angle_per_class=2*np.pi/float(bin_num)

    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # milestones:調整lr的epoch數，gamma:decay factor (https://hackmd.io/@Hong-Jia/H1hmbNr1d)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_SGD, milestones=[i for i in range(10, epochs, 20)], gamma=0.5)

    #dim_loss_func = nn.MSELoss().to(device) #org function
    
    W_consist = 0.3
    W_ry = 0.1

    if is_group:
        print("< Train with GroupLoss >")
        group_loss_func = stdGroupLoss_heading_bin
    
    # Load latest-weights parameters
    weights_folder = FLAGS.weights_path.split('/')[0]
    os.makedirs(weights_folder, exist_ok=True)
    if FLAGS.latest_weights is not None:
        
        weights_path = os.path.join(weights_folder, FLAGS.latest_weights)
        checkpoint = torch.load(weights_path, map_location=device)
        # 如果--bin跟checkpoint['bin']不同會跳錯誤
        assert bin_num == checkpoint['bin'], f'--bin:{bin_num} is not the same as ckpt-bin:{checkpoint["bin"]}'
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        passes = checkpoint['passes']
        best = checkpoint['best']
        #bin_num = checkpoint['bin'] 
        #is_cond = checkpoint['cond'] # for evaluate
        print('Found previous checkpoint: %s at epoch %s'%(weights_path, first_epoch))
        print('Resuming training....')
    else:
        first_epoch = 0
        passes = 0
        best = [0, 0] # epoch, best_mean

    total_num_batches = len(train_loader_L)
    
    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        GT_alpha_list = list()
        pred_alpha_list = list()
        
        for (batch_L, labels_L), (batch_R, labels_R) in zip(train_loader_L, train_loader_R):

            gt_residual = labels_L['heading_residual'].float().to(device)
            gt_bin = labels_L['heading_class'].long().to(device)#這個角度在哪個class上
            gt_dim = labels_L['Dimensions'].float().to(device)
            gt_theta_ray_L = labels_L['Theta_ray'].float().to(device)
            gt_theta_ray_R = labels_L['Theta_ray'].float().to(device)

            batch_L=batch_L.float().to(device)
            batch_R=batch_R.float().to(device)
            model.train()
            [residual_L, bin_L, dim_L] = model(batch_L)

            bin_loss = F.cross_entropy(bin_L, gt_bin,reduction='mean').to(device)
            residual_loss, rediual_val = compute_residual_loss(residual_L, gt_bin, gt_residual, device)
            
            # calc GT_alpha,return list type
            bin_argmax = torch.max(bin_L, dim=1)[1]
            pred_alpha = angle_per_class*bin_argmax + residual_L[torch.arange(len(residual_L)), bin_argmax]
            GT_alpha = angle_per_class*gt_bin + gt_residual
            pred_alpha_list += pred_alpha.tolist()
            GT_alpha_list += GT_alpha.tolist()

            loss_theta = bin_loss + residual_loss
            #dim_loss = F.mse_loss(dim, gt_dim, reduction='mean')  # org use mse_loss
            dim_loss = F.l1_loss(dim_L, gt_dim, reduction='mean')  # 0613 added (monodle, monogup used) (compare L1 vs mse loss)
            #dim_loss = L1_loss_alpha(dim, gt_dim, GT_alpha, device) # 0613 try elevate dim performance            

            loss = alpha * dim_loss + loss_theta
            #added loss
            if is_group and epoch > warm_up:
                truth_Theta = labels_L['Theta'].float().to(device)
                truth_Ry = labels_L['Ry'].float().to(device)
                truth_group = labels_L['Group'].float().to(device)
                group_loss = group_loss_func(pred_alpha, truth_Theta, truth_group, device)
                loss += 0.3 * group_loss
            else:
                group_loss = torch.tensor(0.0).to(device)


            # 0801 added consist loss
            #model.eval()
            reg_ry_L = compute_ry(bin_L, residual_L, gt_theta_ray_L, angle_per_class)
            [residual_R, bin_R, dim_R] = model(batch_R)
            reg_ry_R = compute_ry(bin_R, residual_R, gt_theta_ray_R, angle_per_class)

            ry_angle_loss = F.l1_loss(torch.cos(reg_ry_L), torch.cos(reg_ry_R), reduction='mean')
            consist_loss = F.l1_loss(dim_L, dim_R, reduction='mean')

            loss += W_consist*consist_loss.to(device) + W_ry*ry_angle_loss.to(device)

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            if passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[bin_loss:%.4f],[residual_loss:%.4f],[dim_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(), bin_loss.item(), residual_loss.item(), dim_loss.item()))
                print("[consist_loss: %.4f],[Ry_angle_loss:%.4f]" \
                    %(consist_loss.item(), ry_angle_loss.item()))
                writer.add_scalar('pass/bin_loss', bin_loss, passes//200)
                writer.add_scalar('pass/residual_loss', residual_loss, passes//200)
                writer.add_scalar('pass/dim_loss', dim_loss, passes//200)
                writer.add_scalar('pass/loss_theta', loss_theta, passes//200)
                writer.add_scalar('pass/total_loss', loss, passes//200)
                writer.add_scalar('pass/consist_loss', consist_loss, passes//200)
                writer.add_scalar('pass/ry_angle_loss', ry_angle_loss, passes//200)
                if is_group and epoch > warm_up:
                    print('[group_loss:%.4f]'%(group_loss.item()))
                    writer.add_scalar('pass/group_loss', group_loss, passes//200)

            passes += 1
            curr_batch += 1

        alpha_performance = angle_criterion(pred_alpha_list, GT_alpha_list)
        #print(f'Epoch:{epoch} lr = {scheduler.get_last_lr()[0]}')
        print(f'alpha_performance: {alpha_performance:.4f}') #close to 0 is better
        # record the best_epoch and best_mean
        if alpha_performance < best[1]:
            best = [epoch, alpha_performance]
        writer.add_scalar('epoch/alpha_performance', alpha_performance, epoch)
        #write every epoch
        writer.add_scalar('epoch/bin_loss', bin_loss, epoch)
        writer.add_scalar('epoch/residual_loss', residual_loss, epoch)
        writer.add_scalar('epoch/dim_loss', dim_loss, epoch)
        writer.add_scalar('epoch/loss_theta', loss_theta, epoch)
        writer.add_scalar('epoch/total_loss', loss, epoch) 
        writer.add_scalar('epoch/group_loss', group_loss, epoch)
        writer.add_scalar('epoch/consist_loss', consist_loss, passes//200)
        writer.add_scalar('epoch/ry_angle_loss', ry_angle_loss, passes//200)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        #tensorboard --logdir=./{log_foler} --port 8123

        # save after every 10 epochs
        #scheduler.step()
        if epoch % 10 == 0:
            name = FLAGS.weights_path.split('.')[0] + f'_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print(f'Best mean:{best[1]} @ epoch {best[0]}')
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss,
                    'passes': passes, # for continue training
                    'best': best,
                    'bin': FLAGS.bin, # for evaluate
                    'cond': FLAGS.cond, # for evaluate
                    'W_consist': W_consist,
                    'W_ry': W_ry
                    }, name)
            print("====================")
            
    writer.close()
    #print(f'Elapsed time:{(time.time()-start)//60}min')

def compute_ry(bin, residual, theta_rays, angle_per_class):
    bin_argmax = torch.max(bin, dim=1)[1]
    residual = residual[torch.arange(len(residual)), bin_argmax] 
    alphas = angle_per_class*bin_argmax + residual #mapping bin_class and residual to get alpha
    rys = list()
    for a, ray in zip(alphas, theta_rays):
        rys.append(angle_correction(a+ray))
    return torch.Tensor(rys)

if __name__=='__main__':
    main()
