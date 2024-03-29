from torch_lib.Dataset_heading_bin import *
from torch_lib.ELAN_Dataset import *
from torch_lib.Model_heading_bin import *
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
#path setting
parser.add_argument("--weights-path", "-W_PATH", required=True, help='folder/date ie.weights/0721')
parser.add_argument("--log-dir", default='log', help='tensorboard log-saved path')

#training setting
parser.add_argument("--device", "-D", type=int, default=0, help='select cuda index')
parser.add_argument("--epoch", "-E", type=int, default=50, help='epoch num')

#parser.add_argument("--batch-size", type=int, default=16, help='batch size')

# hyper-parameter (group | bin | cond)
parser.add_argument("--normal", "-N", type=int, default=0, help='0:ImageNet, 1:ELAN')
parser.add_argument("--bin", "-B", type=int, default=4, help='heading bin num')
parser.add_argument("--group", "-G", type=int, help='if True, add stdGroupLoss')
parser.add_argument("--warm-up", "-W", type=int, default=10, help='warm up before adding group loss')
parser.add_argument("--cond", "-C", type=int, default=0, help='if True, 4-dim with theta_ray | boxH_2d ')
parser.add_argument("--aug", "-A", type=int, default=0, help='if True, flip dataset as augmentation')

def main():
    
    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    is_aug = FLAGS.aug
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    normalize_type = FLAGS.normal
    W_group = 0.3

    # make weights folder
    weights_folder = os.path.join('weights', FLAGS.weights_path.split('/')[1])
    os.makedirs(weights_folder, exist_ok=True)
    save_path, log_path, train_config = name_by_parameters(FLAGS)
    os.makedirs(log_path, exist_ok=True)
    print(f'SAVE PATH:{save_path}, LOG PATH:{log_path}, config:{train_config}')
    writer = SummaryWriter(log_path)
    epochs = FLAGS.epoch
    batch_size = 16 #64 worse than 8
    alpha = 0.6

    # model
    train_dataset = ELAN_Dataset('Elan_3d_box', split='train', condition=FLAGS.cond, num_heading_bin=FLAGS.bin, normal=normalize_type)
    train_dataset_flip = ELAN_Dataset('Elan_3d_box', split='train', condition=FLAGS.cond, num_heading_bin=FLAGS.bin, normal=normalize_type, is_flip=True)
    
    print(f"Loading all training files in ELAN dataset:{len(train_dataset)}")
    val_dataset = ELAN_Dataset('Elan_3d_box', split='val', condition=FLAGS.cond, num_heading_bin=FLAGS.bin, normal=normalize_type)
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    train_loader = data.DataLoader(train_dataset, **params)
    train_loader_flip = data.DataLoader(train_dataset_flip, **params)
    valid_loader = data.DataLoader(val_dataset, **params)

    my_vgg = vgg.vgg19_bn(weights='DEFAULT')
    
    if is_cond:
        print("< 4-dim input, Theta_ray as Condition >")
        my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
    
    model = vgg_Model(features=my_vgg.features, bins=bin_num).to(device)
    angle_per_class=2*np.pi/float(bin_num)

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    # milestones:調整lr的epoch數，gamma:decay factor (https://hackmd.io/@Hong-Jia/H1hmbNr1d)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(10, epochs, 20)], gamma=0.5)

    #dim_loss_func = nn.MSELoss().to(device) #org function

    if is_group:
        print("< Train with GroupLoss >")
        #group_loss_func = stdGroupLoss_heading_bin
        group_loss_func = GroupLoss #0801 adjust
    
    passes = 0
    total_num_batches = len(train_loader)
    start = time.time()
    for epoch in range(1, epochs+1):
        curr_batch = 0
        
        model.train()
        for (local_batch, local_labels), (batch_flip, label_flip) in zip(train_loader, train_loader_flip):
            optimizer.zero_grad()

            truth_residual = local_labels['heading_residual'].float().to(device)
            truth_bin = local_labels['heading_class'].long().to(device)#這個角度在哪個class上
            truth_dim = local_labels['Dimensions'].float().to(device)

            truth_dim_flip = label_flip['Dimensions'].float().to(device)
            batch_flip = batch_flip.float().to(device)
            local_batch=local_batch.float().to(device)
            [orient_residual, bin_conf, dim] = model(local_batch)

            bin_loss = F.cross_entropy(bin_conf,truth_bin,reduction='mean').to(device)
            residual_loss = compute_residual_loss(orient_residual,truth_bin,truth_residual, device)
        
            # performance alpha_l1 > l1 > mse
            #dim_loss = F.mse_loss(dim, truth_dim, reduction='mean')  # org use mse_loss
            dim_loss = F.l1_loss(dim, truth_dim, reduction='mean')  # 0613 added (monodle, monogup used) (compare L1 vs mse loss)
            #dim_loss = L1_loss_alpha(dim, truth_dim, GT_alpha, device) # 0613 try elevate dim performance       

            #added loss
            if is_group and epoch > warm_up:
                truth_Theta = local_labels['Theta'].float().to(device)
                #truth_Ry = local_labels['Ry'].float().to(device)
                truth_group = local_labels['Group'].float().to(device)
                bin_argmax = torch.max(bin_conf, dim=1)[1]
                pred_alpha = angle_per_class*bin_argmax + orient_residual[torch.arange(len(orient_residual)), bin_argmax]
                #group_loss = group_loss_func(pred_alpha, truth_Theta, truth_group, device)
                #(orient_batch, orientGT_batch, confGT_batch, group_batch, device)
                group_loss = group_loss_func(pred_alpha, truth_Theta, truth_group, device)
            else:
                group_loss = torch.tensor(0.0).to(device)
            
            loss = alpha * dim_loss + bin_loss + residual_loss + W_group * group_loss # W_group=0.3 before0724
            
            # 0831 added
            with torch.no_grad():
                [_, _, dim_flip] = model(batch_flip)
                flip_loss = F.l1_loss(dim, dim_flip, reduction='mean')
            
            loss += 1 * flip_loss

            loss.backward()
            optimizer.step()

            if passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[bin_loss:%.4f],[residual_loss:%.4f],[dim_loss:%.4f],[flip_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(), bin_loss.item(), residual_loss.item(), dim_loss.item(), flip_loss.item()))
                if is_group and epoch>warm_up:
                    print("[group_loss:%.4f]"%(group_loss.item()))
            
            passes += 1
            curr_batch += 1

        # eval part
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                GT_alpha_list = list()
                REG_alpha_list = list()

                GT_dim_list = list()
                REG_dim_list = list()
                
                for batch, labels in valid_loader:
                    
                    gt_residual = labels['heading_residual'].float().to(device)
                    gt_bin = labels['heading_class'].long().to(device)#這個角度在哪個class上
                    gt_dim = labels['Dimensions'].float().to(device)

                    batch=batch.float().to(device)
                    [residual, bin, dim] = model(batch)
                    # calc GT_alpha,return list type
                    bin_argmax = torch.max(bin, dim=1)[1]
                    reg_alpha = angle_per_class*bin_argmax + residual[torch.arange(len(residual)), bin_argmax]
                    GT_alphas = angle_per_class*gt_bin + gt_residual

                    REG_alpha_list += reg_alpha.cpu().tolist()
                    GT_alpha_list += GT_alphas.cpu().tolist()

                    REG_dim_list += dim.cpu().tolist()
                    GT_dim_list += gt_dim.cpu().tolist()
                
                alpha_performance = angle_criterion(REG_alpha_list, GT_alpha_list)
                GT_dim_list = np.array(GT_dim_list)
                REG_dim_list = np.array(REG_dim_list)
                dim_performance =  np.mean(abs(GT_dim_list-REG_dim_list), axis=0)
                print(f'[Alpha diff]: {alpha_performance:.4f}') #close to 0 is better
                print(f'[DIM diff] H:{dim_performance[0]:.4f}, W:{dim_performance[1]:.4f}, L:{dim_performance[2]:.4f}')
                writer.add_scalar(f'{train_config}/alpha_performance', alpha_performance, epoch)
                writer.add_scalar(f'{train_config}/H', dim_performance[0], epoch)
                writer.add_scalar(f'{train_config}/W', dim_performance[1], epoch)
                writer.add_scalar(f'{train_config}/L', dim_performance[2], epoch)

        #write every epoch
        writer.add_scalar(f'{train_config}/bin_loss', bin_loss, epoch)
        writer.add_scalar(f'{train_config}/residual_loss', residual_loss, epoch)
        writer.add_scalar(f'{train_config}/dim_loss', dim_loss, epoch)
        writer.add_scalar(f'{train_config}/total_loss', loss, epoch)
        writer.add_scalar(f'{train_config}/group_loss', group_loss, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        # MobaXterm https://zhuanlan.zhihu.com/p/138811263
        #tensorboard --logdir=./{log_foler} --port 8123

        # save after every 10 epochs
        #scheduler.step()
        if epoch % epochs == 0:
            name = save_path + f'_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'bin': bin_num, # for evaluate
                    'cond': is_cond, # for evaluate
                    'normal': normalize_type,
                    'W_group': W_group
                    }, name)
            print("====================")
    writer.close()
    print(f'Elapsed time:{(time.time()-start)//60}min')

def name_by_parameters(FLAGS):
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    is_aug = FLAGS.aug
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    normalize_type = FLAGS.normal
    
    save_path = f'{FLAGS.weights_path}_B{bin_num}_N{normalize_type}'
    if is_group==1:
        save_path += f'_G_W{warm_up}'
    if is_cond==1:
        save_path += '_C'
    if is_aug==1:
        save_path += '_aug'

    train_config = save_path.split("weights/")[1]
    log_path = f'log/{train_config}'

    return save_path, log_path, train_config

if __name__=='__main__':
    main()
