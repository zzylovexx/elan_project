from torch_lib.KITTI_Dataset_0808 import *
from torch_lib.Model_heading_bin import Model, compute_residual_loss
from library.ron_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg
from torch.utils import data
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=2023, help='keep seeds to represent same result')
#path setting
parser.add_argument("--weights-path", "-W_PATH", required=True, help='folder/date ie.weights/0721')

#training setting
parser.add_argument("--type", "-T", type=int, default=0, help='0:dim, 1:alpha, 2:both, 3:BL')
parser.add_argument("--device", "-D", type=int, default=0, help='select cuda index')
parser.add_argument("--epoch", "-E", type=int, default=50, help='epoch num')

#parser.add_argument("--batch-size", type=int, default=16, help='batch size')

# hyper-parameter (group | bin | cond)
parser.add_argument("--bin", "-B", type=int, default=4, help='heading bin num')
parser.add_argument("--group", "-G", type=int, help='if True, add stdGroupLoss')
parser.add_argument("--warm-up", "-W", type=int, default=10, help='warm up before adding group loss')
parser.add_argument("--cond", "-C", type=int, help='if True, 4-dim with theta_ray | boxH_2d ')
# TO BE ADDED (Loss的weights比例alpha, w of groupLoss, LRscheduler: milestone, gamma )

def main():
    cfg = {'path':'Kitti/training',
            'class_list':['car'], 'diff_list': [1,2], #0:DontCare, 1:Easy, 2:Moderate, 3:Hard, 4:Unknown
            'bins': 4, 'cond':False}
    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    type = FLAGS.type
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    
    epochs = FLAGS.epoch
    batch_size = 16 #64 worse than 8
    alpha = 0.6
    W_consist = 0.3
    W_ry = 0.1
    # make weights folder
    weights_folder = os.path.join('weights', FLAGS.weights_path.split('/')[1])
    os.makedirs(weights_folder, exist_ok=True)
    save_path, log_path, train_config = name_by_parameters(FLAGS)
    print(f'SAVE PATH:{save_path}, LOG PATH:{log_path}, config:{train_config}')
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    

    # model
    print("Loading all detected objects in dataset...")
    print('Kitti dataset')
    process = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize([224,224], transforms.InterpolationMode.BICUBIC), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = KITTI_Dataset(cfg, process, split='train')
    dataset_valid = KITTI_Dataset(cfg, process, split='val')
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    train_loader = data.DataLoader(dataset_train, **params)
    valid_loader = data.DataLoader(dataset_valid, **params)

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
        group_loss_func = stdGroupLoss_heading_bin
    
    total_num_batches = len(train_loader)
    passes = 0
    for epoch in range(1, epochs+1):
        curr_batch = 0
        
        for batch_L, labels_L, batch_R, labels_R in train_loader:

            gt_residual = labels_L['Heading_res'].float().to(device)
            gt_bin = labels_L['Heading_bin'].long().to(device)#這個角度在哪個class上
            gt_dim = labels_L['Dim_delta'].float().to(device)
            gt_theta_ray_L = labels_L['Theta_ray'].float().to(device)
            gt_theta_ray_R = labels_L['Theta_ray'].float().to(device)

            batch_L=batch_L.float().to(device)
            batch_R=batch_R.float().to(device)
            model.train()
            [residual_L, bin_L, dim_L] = model(batch_L)

            bin_loss = F.cross_entropy(bin_L, gt_bin,reduction='mean').to(device)
            residual_loss, _ = compute_residual_loss(residual_L, gt_bin, gt_residual, device)
            
            # calc GT_alpha,return list type
            bin_argmax = torch.max(bin_L, dim=1)[1]
            

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
                pred_alpha = angle_per_class*bin_argmax + residual_L[torch.arange(len(residual_L)), bin_argmax]
                group_loss = group_loss_func(pred_alpha.detach(), truth_Theta, truth_group, device)
                loss += 0.3 * group_loss
            else:
                group_loss = torch.tensor(0.0).to(device)


            # 0801 added consist loss
            if type!= 3: # baseline
                reg_ry_L = compute_ry(bin_L, residual_L, gt_theta_ray_L, angle_per_class)
                [residual_R, bin_R, dim_R] = model(batch_R)
                reg_ry_R = compute_ry(bin_R, residual_R, gt_theta_ray_R, angle_per_class)
                consist_loss = F.l1_loss(dim_L, dim_R, reduction='mean')
                ry_angle_loss = F.l1_loss(torch.cos(reg_ry_L), torch.cos(reg_ry_R), reduction='mean')
                
                # angle_consist
                if type==0:
                    ry_angle_loss = torch.tensor(0.0)
                # dim_consist
                elif type==1:
                    consist_loss = torch.tensor(0.0)
                # if type==2 : both calculated

                loss += W_consist*consist_loss.to(device) + W_ry*ry_angle_loss.to(device)

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            if passes % 200 == 0 and type!=3:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[bin_loss:%.4f],[residual_loss:%.4f],[dim_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(), bin_loss.item(), residual_loss.item(), dim_loss.item()))
                print("[consist_loss: %.4f],[Ry_angle_loss:%.4f]" \
                    %(consist_loss.item(), ry_angle_loss.item()))
                if is_group and epoch > warm_up:
                    print('[group_loss:%.4f]'%(group_loss.item()))
            
            elif passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[bin_loss:%.4f],[residual_loss:%.4f],[dim_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(), bin_loss.item(), residual_loss.item(), dim_loss.item()))
                if is_group and epoch > warm_up:
                    print('[group_loss:%.4f]'%(group_loss.item()))

            passes += 1
            curr_batch += 1

        #print(f'Epoch:{epoch} lr = {scheduler.get_last_lr()[0]}')
        
        #write every epoch
        writer.add_scalar(f'{train_config}/bin_loss', bin_loss, epoch)
        writer.add_scalar(f'{train_config}/residual_loss', residual_loss, epoch)
        writer.add_scalar(f'{train_config}/dim_loss', dim_loss, epoch)
        writer.add_scalar(f'{train_config}/loss_theta', loss_theta, epoch)
        writer.add_scalar(f'{train_config}/total_loss', loss, epoch) 
        writer.add_scalar(f'{train_config}/group_loss', group_loss, epoch)
        if type!=3:
            writer.add_scalar(f'{train_config}/consist_loss', consist_loss, epoch)
            writer.add_scalar(f'{train_config}/ry_angle_loss', ry_angle_loss, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        #tensorboard --logdir=./{log_foler} --port 8123

        # save after every 10 epochs
        #scheduler.step()
        
        model.eval()
        with torch.no_grad():
            GT_alpha_list = list()
            REG_alpha_list = list()

            GT_dim_list = list()
            REG_dim_list = list()
            
            for batch, labels in valid_loader:
                
                gt_residual = labels['Heading_res'].float().to(device)
                gt_bin = labels['Heading_bin'].long().to(device)#這個角度在哪個class上
                gt_dim = labels['Dim_delta'].float().to(device)
                gt_theta_ray_L = labels['Theta_ray'].float().to(device)

                batch=batch.float().to(device)
                [residual, bin, dim] = model(batch)
                # calc GT_alpha,return list type
                bin_argmax = torch.max(bin, dim=1)[1]
                reg_alpha = angle_per_class*bin_argmax + residual[torch.arange(len(residual)), bin_argmax]
                GT_alpha = angle_per_class*gt_bin + gt_residual

                REG_alpha_list += reg_alpha.cpu().tolist()
                GT_alpha_list += GT_alpha.cpu().tolist()

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
                

        if epoch % epochs == 0:
            name = save_path + f'_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss,
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


def name_by_parameters(FLAGS):
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    
    save_path = f'{FLAGS.weights_path}_B{bin_num}'
    if is_group==1:
        save_path += f'_G_W{warm_up}'
    if is_cond==1:
        save_path += '_C'

    train_config = save_path.split("weights/")[1]
    log_path = f'log/{train_config}'

    return save_path, log_path, train_config

if __name__=='__main__':
    main()
