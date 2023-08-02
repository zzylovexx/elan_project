from torch_lib.Dataset_heading_bin import *
from torch_lib.ELAN_Dataset import *
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
#path setting
parser.add_argument("--weights-path", "-W_PATH", required=True, help='folder/date ie.weights/0721')
parser.add_argument("--latest-weights", default=None, help='only input the weights-name.pkl') #in the same folder as above
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
parser.add_argument("--cond", "-C", type=int, help='if True, 4-dim with theta_ray | boxH_2d ')

def main():
    
    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    normalize_type = FLAGS.normal
    W_group = 0.3
    save_path = f'{FLAGS.weights_path}_B{bin_num}_N{normalize_type}'
    if is_group:
        save_path += f'_G_W{warm_up}'
    if is_cond:
        save_path += '_C'

    train_config = save_path.split("weights/")[1]
    log_path = os.path.join(FLAGS.log_dir, train_config)
    os.makedirs(log_path, exist_ok=True)
    print(f'SAVE PATH:{save_path}, LOG PATH:{log_path}, config:{train_config}')
    writer = SummaryWriter(log_path)
    
    writer = SummaryWriter(FLAGS.log_dir)
    epochs = FLAGS.epoch
    batch_size = 16 #64 worse than 8
    alpha = 0.6

    # model
    train_dataset = ELAN_Dataset('Elan_3d_box', split='train', condition=FLAGS.cond, num_heading_bin=FLAGS.bin, normal=normalize_type)
    #train_dataset = ELAN_Dataset('Elan_3d_box', split='trainval', condition=FLAGS.cond, num_heading_bin=FLAGS.bin, normal=normalize_type)
    print(f"Loading all training files in ELAN dataset:{len(train_dataset.ids)}")

    val_dataset = ELAN_Dataset('Elan_3d_box', split='val', condition=FLAGS.cond, num_heading_bin=FLAGS.bin, normal=normalize_type)
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    train_loader = data.DataLoader(train_dataset, **params)
    val_loader = data.DataLoader(val_dataset, **params)

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

    if is_group:
        print("< Train with GroupLoss >")
        #group_loss_func = stdGroupLoss_heading_bin
        group_loss_func = GroupLoss #0801 adjust
    
    # Load latest-weights parameters
    weights_folder = FLAGS.weights_path.split('/')[0] + '/' + FLAGS.weights_path.split('/')[1]
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
        #bin_num = checkpoint['bin'] 
        #is_cond = checkpoint['cond'] # for evaluate
        print('Found previous checkpoint: %s at epoch %s'%(weights_path, first_epoch))
        print('Resuming training....')
    else:
        first_epoch = 0
        passes = 0

    total_num_batches = len(train_loader)
    best = [0, 1] # best_epoch, best_alpha_performance
    start = time.time()
    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        model.train()
        for local_batch, local_labels in train_loader:

            truth_residual = local_labels['heading_residual'].float().to(device)
            truth_bin = local_labels['heading_class'].long().to(device)#這個角度在哪個class上
            truth_dim = local_labels['Dimensions'].float().to(device)

            local_batch=local_batch.float().to(device)
            [orient_residual, bin_conf, dim] = model(local_batch)

            bin_loss = F.cross_entropy(bin_conf,truth_bin,reduction='mean').to(device)
            residual_loss, _ = compute_residual_loss(orient_residual,truth_bin,truth_residual, device)
        
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
            
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            if passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[bin_loss:%.4f],[residual_loss:%.4f],[dim_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(), bin_loss.item(), residual_loss.item(), dim_loss.item()))
                writer.add_scalar('pass/bin_loss', bin_loss, passes//200)
                writer.add_scalar('pass/residual_loss', residual_loss, passes//200)
                writer.add_scalar('pass/dim_loss', dim_loss, passes//200)
                writer.add_scalar('pass/total_loss', loss, passes//200) 

                if is_group and epoch>warm_up:
                    print("[group_loss:%.4f]"%(group_loss.item()))
                    writer.add_scalar('pass/group_loss', group_loss, passes//200)
            
            passes += 1
            curr_batch += 1
        '''
        # eval part
        model.eval()
        GT_alpha_list = list()
        pred_alpha_list = list()
        for local_batch, local_labels in val_loader:

            truth_residual = local_labels['heading_residual'].float().to(device)
            truth_bin = local_labels['heading_class'].long().to(device)#這個角度在哪個class上
            truth_dim = local_labels['Dimensions'].float().to(device)

            local_batch=local_batch.float().to(device)

            [orient_residual, bin_conf, dim] = model(local_batch)
            # calc GT_alpha,return list type
            bin_argmax = torch.max(bin_conf, dim=1)[1]
            pred_alpha = angle_per_class*bin_argmax + orient_residual[torch.arange(len(orient_residual)), bin_argmax]
            GT_alpha = angle_per_class*truth_bin + truth_residual
            pred_alpha_list += pred_alpha.tolist()
            GT_alpha_list += GT_alpha.tolist()

        alpha_performance = angle_criterion(pred_alpha_list, GT_alpha_list)
        print(f'alpha_performance: {alpha_performance:.4f}') #close to 0 is better\
        writer.add_scalar('epoch/alpha_performance', alpha_performance, epoch)
        if alpha_performance < best[1]:
            best[0] = epoch
            best[1] = alpha_performance
        '''
        #write every epoch
        writer.add_scalar('epoch/bin_loss', bin_loss, epoch)
        writer.add_scalar('epoch/residual_loss', residual_loss, epoch)
        writer.add_scalar('epoch/dim_loss', dim_loss, epoch)
        writer.add_scalar('epoch/total_loss', loss, epoch)
        writer.add_scalar('epoch/group_loss', group_loss, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        # MobaXterm https://zhuanlan.zhihu.com/p/138811263
        #tensorboard --logdir=./{log_foler} --port 8123

        # save after every 10 epochs
        #scheduler.step()
        if epoch % (epochs//2) == 0:
            name = save_path + f'_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print(f'Best alpha performance:{best[1]} @ epoch {best[0]}')
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss,
                    'passes': passes, # for continue training
                    'bin': bin_num, # for evaluate
                    'cond': is_cond, # for evaluate
                    'normal': normalize_type,
                    'W_group': W_group
                    }, name)
            print("====================")
    writer.close()
    print(f'Elapsed time:{(time.time()-start)//60}min')

if __name__=='__main__':
    main()
