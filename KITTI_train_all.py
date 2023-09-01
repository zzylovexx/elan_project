from torch_lib.KITTI_Dataset import *
from torch_lib.Model_heading_bin import *
from library.ron_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg, resnet, densenet
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
parser.add_argument("--network", "-N", type=int, default=0, help='vgg/resnet/densenet')
parser.add_argument("--type", "-T", type=int, default=0, help='0:dim, 1:alpha, 2:both, 3:BL')
parser.add_argument("--device", "-D", type=int, default=0, help='select cuda index')
parser.add_argument("--epoch", "-E", type=int, default=50, help='epoch num')

#parser.add_argument("--batch-size", type=int, default=16, help='batch size')

# hyper-parameter (group | bin | cond)
parser.add_argument("--bin", "-B", type=int, default=4, help='heading bin num')
parser.add_argument("--group", "-G", type=int, help='if True, add stdGroupLoss')
parser.add_argument("--warm-up", "-W", type=int, default=10, help='warm up before adding group loss')
parser.add_argument("--cond", "-C", type=int, help='if True, 4-dim with theta_ray | boxH_2d ')
parser.add_argument("--aug", "-A", type=int, default=0, help='if True, flip dataset as augmentation')
# TO BE ADDED (Loss的weights比例alpha, w of groupLoss, LRscheduler: milestone, gamma )

def main():
    cfg = {'path':'Kitti/training',
            'class_list':['car'], 'diff_list': [1, 2], #0:DontCare, 1:Easy, 2:Moderate, 3:Hard, 4:Unknown
            'bins': 0, 'cond':False, 'group':False, 'network':0}
    '''
    bin_num = 4
    is_cond = 0
    is_group = 3
    warm_up = 0
    epochs=10
    device = torch.device('cuda:0')
    type_=3
    '''
    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    is_aug = FLAGS.aug
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    type_ = FLAGS.type
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    epochs = FLAGS.epoch
    network = FLAGS.network
    batch_size = 16 #64 worse than 8
    W_dim = 1 #0.1~0.14
    W_theta = 1 #0.03~1
    W_group = 0.6 # 0.02
    W_consist = 1 #數值小0.02~0.04  TODO W_consist要調高(0818) tried bad:3,5
    W_ry = 0.1 #數值大0.05~0.2
    W_depth = 0.05 # 2
    # make weights folder
    cfg['bins'] = bin_num
    cfg['cond'] = is_cond
    cfg['group'] = is_group
    cfg['network'] = network
    
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
    if is_aug:
        dataset_train_flip = KITTI_Dataset(cfg, process, split='train', is_flip=True)
        dataset_train_all = data.ConcatDataset([dataset_train, dataset_train_flip])
    else:
        dataset_train_all = dataset_train
    train_loader = data.DataLoader(dataset_train_all, **params)
    valid_loader = data.DataLoader(dataset_valid, **params)

    # 0818 added
    if network==0:
        my_vgg = vgg.vgg19_bn(weights='DEFAULT') #512x7x7
        model = vgg_Model(features=my_vgg.features, bins=bin_num).to(device)
    elif network==1:
        my_resnet = resnet.resnet18(weights='DEFAULT')
        my_resnet = torch.nn.Sequential(*(list(my_resnet.children())[:-2])) #512x7x7
        model = resnet_Model(features=my_resnet, bins=bin_num).to(device) # resnet no features
    elif network==2:
        my_dense = densenet.densenet121(weights='DEFAULT') #1024x7x7
        model = dense_Model(features=my_dense.features, bins=bin_num).to(device)

    if is_cond:
        print("< 4-dim input, Theta_ray as Condition >")
        model.features[0].in_channels = 4
        #model.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))

    angle_per_class=2*np.pi/float(bin_num)

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    # milestones:調整lr的epoch數，gamma:decay factor (https://hackmd.io/@Hong-Jia/H1hmbNr1d)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(10, epochs, 20)], gamma=0.5)
    #dim_loss_func = nn.MSELoss().to(device) #org function
    if is_group == 1:
        print("< Train with compute_cos_group_loss >")
        group_loss_func = compute_cos_group_loss
    elif is_group ==2 :
        print("< Train with compute_sin_sin_group_loss >")
        group_loss_func = compute_sin_sin_group_loss
    elif is_group == 3:
        print("< Train with compute_compare_group_loss >")
        group_loss_func = compute_compare_group_loss

    
    total_batches = len(train_loader)
    start = time.time()
    print('total_batches', total_batches)
    for epoch in range(1, epochs+1):
        avg_bin_loss, avg_residual_loss, avg_theta_loss = 0, 0, 0
        avg_dim_loss, avg_consist_loss, avg_angle_loss, avg_depth_loss = 0,0,0,0
        avg_total_loss = 0
        model.train()
        for batch_L, labels_L, batch_R, labels_R in train_loader:
            optimizer.zero_grad()

            gt_residual = labels_L['Heading_res'].float().to(device)
            gt_bin = labels_L['Heading_bin'].long().to(device)#這個角度在哪個class上
            gt_dim = labels_L['Dim_delta'].float().to(device)
            gt_theta_ray_L = labels_L['Theta_ray'].float().to(device)
            gt_theta_ray_R = labels_R['Theta_ray'].float().to(device)
            gt_depth = labels_L['Depth'].float().to(device)
            gt_img_W = labels_L['img_W']
            gt_box2d = labels_L['Box2d']
            gt_calib = labels_L['Calib']
            gt_class = labels_L['Class']

            batch_L=batch_L.float().to(device)
            batch_R=batch_R.float().to(device)
            [residual_L, bin_L, dim_L] = model(batch_L)

            bin_loss = W_theta * F.cross_entropy(bin_L, gt_bin, reduction='mean').to(device)
            residual_loss = W_theta * compute_residual_loss(residual_L, gt_bin, gt_residual, device)
            loss_theta = bin_loss + residual_loss
            
            
            GT_alphas = labels_L['Alpha'].to(device)
            #dim_loss = F.mse_loss(dim, gt_dim, reduction='mean')  # org use mse_loss
            dim_loss = W_dim * F.l1_loss(dim_L, gt_dim, reduction='mean')  # 0613 added (monodle, monogup used) (compare L1 vs mse loss)
            #dim_loss = W_dim * L1_loss_alpha(dim_L, gt_dim, GT_alphas, device) # 0613 try elevate dim performance       
            #dim_loss = W_dim * F.mse_loss(dim_L, gt_dim, reduction='mean').to(device) # 0613 try elevate dim performance

            loss = dim_loss + loss_theta
            #added loss
            if is_group > 0 and epoch > warm_up:
                # before 0814 group_alpha_loss
                #REG_alphas = compute_alpha(bin_L, residual_L, angle_per_class).to(device)
                #group_loss = group_loss_func(REG_alphas, GT_alphas)
                # 0815 group_ry_loss
                GT_rys = labels_L['Ry'].to(device)
                REG_rys = compute_ry(bin_L, residual_L, gt_theta_ray_L, angle_per_class).to(device)
                group_loss = W_group * group_loss_func(REG_rys, GT_rys).to(device)
                loss += group_loss
            else:
                group_loss = torch.tensor(0.0).to(device)


            # 0801 added consist loss
            if type_!= 3: # baseline
                reg_ry_L = compute_ry(bin_L, residual_L, gt_theta_ray_L, angle_per_class)
                with torch.no_grad(): #0817 added
                    [residual_R, bin_R, dim_R] = model(batch_R)
                    reg_ry_R = compute_ry(bin_R, residual_R, gt_theta_ray_R, angle_per_class)
                    consist_loss = W_consist * F.l1_loss(dim_L, dim_R, reduction='mean').to(device)
                    ry_angle_loss = W_ry * F.l1_loss(torch.cos(reg_ry_L), torch.cos(reg_ry_R), reduction='mean').to(device)
                # angle_consist
                if type_==0:
                    ry_angle_loss = torch.tensor(0.0).to(device)
                # dim_consist
                elif type_==1:
                    consist_loss = torch.tensor(0.0).to(device)
                # if type_==2 : both calculated

                loss += consist_loss + ry_angle_loss
        
            # depth_loss
            calc_depth = list()
            for i in range(batch_L.shape[0]):
                img_W = gt_img_W[i]
                box2d = gt_box2d[i]
                cam_to_img = gt_calib[i]
                obj_W = dataset_train.get_cls_dim_avg(gt_class[i])[1] + dim_L.cpu().detach().numpy()[i][1]
                obj_L = dataset_train.get_cls_dim_avg(gt_class[i])[2] + dim_L.cpu().detach().numpy()[i][2]
                #alpha = reg_alpha[i]
                alpha = GT_alphas[i].cpu().detach()
                calc_depth.append(calc_depth_with_alpha_theta(img_W, box2d, cam_to_img, obj_W, obj_L, alpha, trun=0.0))
            calc_depth = torch.FloatTensor(calc_depth).to(device)
            depth_loss = W_depth * F.l1_loss(gt_depth, calc_depth).to(device)
            loss += depth_loss 
            
            loss.backward()
            optimizer.step()

            avg_bin_loss += bin_loss.item()*len(batch_L)
            avg_residual_loss += residual_loss.item()*len(batch_L)
            avg_theta_loss += loss_theta.item()*len(batch_L)
            avg_dim_loss += dim_loss.item()*len(batch_L)
            avg_depth_loss += depth_loss.item()*len(batch_L) # my
            avg_total_loss += loss.item()*len(batch_L)
            if type_!= 3:
                avg_consist_loss += consist_loss.item()*len(batch_L) # my
                avg_angle_loss += ry_angle_loss.item()*len(batch_L) # my
            

        avg_bin_loss/=len(dataset_train_all)
        avg_residual_loss/=len(dataset_train_all)
        avg_theta_loss/=len(dataset_train_all)
        avg_dim_loss/=len(dataset_train_all)
        avg_depth_loss/=len(dataset_train_all)
        avg_total_loss/=len(dataset_train_all)
        if type_!=3:
            avg_consist_loss/=len(dataset_train_all)
            avg_angle_loss/=len(dataset_train_all)

        #print(f'Epoch:{epoch} lr = {scheduler.get_last_lr()[0]}')
        
        if type_!=3:
            print("--- epoch %s --- [loss: %.4f],[theta_loss:%.4f],[dim_loss:%.4f],[depth_loss:%.4f]" \
                %(epoch, avg_total_loss, avg_theta_loss, avg_dim_loss, avg_depth_loss))
            print("[consist_loss: %.4f],[Ry_angle_loss:%.4f]" \
                %(avg_consist_loss, avg_angle_loss))
            if is_group > 0 and epoch > warm_up:
                print('[group_loss:%.4f]'%(group_loss.item()))
        #baseline
        else:
            print("--- epoch %s --- [loss: %.4f],[theta_loss:%.4f],[dim_loss:%.4f],[depth_loss:%.4f]" \
                %(epoch, loss.item(), loss_theta.item(), dim_loss.item(), depth_loss.item()))
            if is_group > 0 and epoch > warm_up:
                print('[group_loss:%.4f]'%(group_loss.item()))

        # save after every 10 epochs
        #scheduler.step()
        
        model.eval()
        with torch.no_grad():
            eval_bin_loss, eval_residual_loss, eval_theta_loss = 0,0,0
            eval_dim_loss, eval_consist_loss, eval_angle_loss, eval_depth_loss = 0,0,0,0
            eval_total_loss = 0
            GT_alpha_list = list()
            REG_alpha_list = list()
            GT_dim_list = list()
            REG_dim_list = list()
            for batch_L, labels_L, batch_R, labels_R in valid_loader:
                gt_residual = labels_L['Heading_res'].float().to(device)
                gt_bin = labels_L['Heading_bin'].long().to(device)#這個角度在哪個class上
                gt_dim = labels_L['Dim_delta'].float().to(device)
                gt_theta_ray_L = labels_L['Theta_ray'].float().to(device)
                gt_theta_ray_R = labels_R['Theta_ray'].float().to(device)
                gt_depth = labels_L['Depth'].float().to(device)
                gt_img_W = labels_L['img_W']
                gt_box2d = labels_L['Box2d']
                gt_calib = labels_L['Calib']
                gt_class = labels_L['Class']

                batch_L=batch_L.float().to(device)
                batch_R=batch_R.float().to(device)
                [residual_L, bin_L, dim_L] = model(batch_L)

                bin_loss = W_theta * F.cross_entropy(bin_L, gt_bin, reduction='mean').to(device)
                residual_loss = W_theta * compute_residual_loss(residual_L, gt_bin, gt_residual, device)
                loss_theta = bin_loss + residual_loss
                
                GT_alphas = labels_L['Alpha'].to(device)
                #dim_loss = F.mse_loss(dim, gt_dim, reduction='mean')  # org use mse_loss
                dim_loss = W_dim * F.l1_loss(dim_L, gt_dim, reduction='mean')  # 0613 added (monodle, monogup used) (compare L1 vs mse loss)
                #dim_loss = W_dim * L1_loss_alpha(dim_L, gt_dim, GT_alphas, device) # 0613 try elevate dim performance       
                #dim_loss = W_dim * F.mse_loss(dim_L, gt_dim, reduction='mean').to(device) # 0613 try elevate dim performance

                loss = dim_loss + loss_theta
                #added loss
                if is_group > 0 and epoch > warm_up:
                    # before 0814 group_alpha_loss
                    #REG_alphas = compute_alpha(bin_L, residual_L, angle_per_class).to(device)
                    #group_loss = group_loss_func(REG_alphas, GT_alphas)
                    # 0815 group_ry_loss
                    GT_rys = labels_L['Ry'].to(device)
                    REG_rys = compute_ry(bin_L, residual_L, gt_theta_ray_L, angle_per_class).to(device)
                    group_loss = W_group * group_loss_func(REG_rys, GT_rys).to(device)
                    loss += group_loss
                else:
                    group_loss = torch.tensor(0.0).to(device)


                # 0801 added consist loss
                if type_!= 3: # baseline
                    reg_ry_L = compute_ry(bin_L, residual_L, gt_theta_ray_L, angle_per_class)
                    [residual_R, bin_R, dim_R] = model(batch_R)
                    reg_ry_R = compute_ry(bin_R, residual_R, gt_theta_ray_R, angle_per_class)
                    consist_loss = W_consist * F.l1_loss(dim_L, dim_R, reduction='mean').to(device)
                    ry_angle_loss = W_ry * F.l1_loss(torch.cos(reg_ry_L), torch.cos(reg_ry_R), reduction='mean').to(device)
                    # angle_consist
                    if type_==0:
                        ry_angle_loss = torch.tensor(0.0).to(device)
                    # dim_consist
                    elif type_==1:
                        consist_loss = torch.tensor(0.0).to(device)
                    # if type_==2 : both calculated
                    eval_consist_loss += consist_loss*len(batch_L)
                    eval_angle_loss += ry_angle_loss*len(batch_L)
                    loss += consist_loss + ry_angle_loss
            
                # depth_loss
                calc_depth = list()
                for i in range(batch_L.shape[0]):
                    img_W = gt_img_W[i]
                    box2d = gt_box2d[i]
                    cam_to_img = gt_calib[i]
                    obj_W = dataset_train.get_cls_dim_avg(gt_class[i])[1] + dim_L.cpu().detach().numpy()[i][1]
                    obj_L = dataset_train.get_cls_dim_avg(gt_class[i])[2] + dim_L.cpu().detach().numpy()[i][2]
                    #alpha = reg_alpha[i]
                    alpha = GT_alphas[i].cpu().detach()
                    calc_depth.append(calc_depth_with_alpha_theta(img_W, box2d, cam_to_img, obj_W, obj_L, alpha, trun=0.0))
                calc_depth = torch.FloatTensor(calc_depth).to(device)
                depth_loss = W_depth * F.l1_loss(gt_depth, calc_depth).to(device)
                loss += depth_loss 
                # get regressed values
                bin_argmax = torch.max(bin_L, dim=1)[1]
                reg_alpha = angle_per_class*bin_argmax + residual_L[torch.arange(len(residual_L)), bin_argmax]
                GT_alphas = angle_per_class*gt_bin + gt_residual
                REG_alpha_list += reg_alpha.cpu().tolist()
                GT_alpha_list += GT_alphas.cpu().tolist()
                REG_dim_list += dim_L.cpu().tolist()
                GT_dim_list += gt_dim.cpu().tolist()
                # sum loss
                eval_bin_loss += bin_loss.item()*len(batch_L)
                eval_residual_loss += residual_loss.item()*len(batch_L)
                eval_theta_loss += loss_theta.item()*len(batch_L)
                eval_dim_loss += dim_loss.item()*len(batch_L)
                eval_depth_loss += depth_loss.item()*len(batch_L) # my
                eval_total_loss += loss.item()*len(batch_L)
                if type_!= 3:
                    eval_consist_loss += consist_loss.item()*len(batch_L) # my
                    eval_angle_loss += ry_angle_loss.item()*len(batch_L) # my
                

            eval_bin_loss/=len(dataset_valid)
            eval_residual_loss/=len(dataset_valid)
            eval_dim_loss/=len(dataset_valid)
            eval_depth_loss/=len(dataset_valid)
            eval_total_loss/=len(dataset_valid)
            if type_!=3:
                eval_consist_loss/=len(dataset_valid)
                eval_angle_loss/=len(dataset_valid)

            
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
        writer.add_scalars(f'{train_config}/bin_loss', {'Train': avg_bin_loss, 'Valid': eval_bin_loss}, epoch)
        writer.add_scalars(f'{train_config}/residual_loss', {'Train': avg_residual_loss, 'Valid': eval_residual_loss}, epoch)
        writer.add_scalars(f'{train_config}/dim_loss', {'Train': avg_dim_loss, 'Valid': eval_dim_loss}, epoch)
        writer.add_scalars(f'{train_config}/loss_theta', {'Train': avg_theta_loss, 'Valid': eval_theta_loss}, epoch)
        writer.add_scalars(f'{train_config}/total_loss', {'Train': avg_total_loss, 'Valid': eval_total_loss}, epoch)
        if is_group:
            writer.add_scalar(f'{train_config}/group_loss', group_loss, epoch)
        if type_!=3:
            writer.add_scalars(f'{train_config}/consist_loss', {'Train': avg_consist_loss, 'Valid': eval_consist_loss}, epoch)
            writer.add_scalars(f'{train_config}/ry_angle_loss', {'Train': avg_angle_loss, 'Valid': eval_angle_loss}, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        #tensorboard --logdir=./{log_foler} --port 8123
            
        if epoch % epochs == 0:
            name = save_path + f'_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cfg': cfg,
                    'W_dim': W_dim,
                    'W_theta': W_theta,
                    'W_consist': W_consist,
                    'W_ry': W_ry,
                    'W_group': W_group,
                    }, name)
            print("====================")
            
    writer.close()
    print(f'Elapsed time:{(time.time()-start)//60}min')

def compute_ry(bin_, residual, theta_rays, angle_per_class):
    bin_argmax = torch.max(bin_, dim=1)[1]
    residual = residual[torch.arange(len(residual)), bin_argmax] 
    alphas = angle_per_class*bin_argmax + residual #mapping bin_class and residual to get alpha
    rys = list()
    for a, ray in zip(alphas, theta_rays):
        rys.append(angle_correction(a+ray))
    return torch.Tensor(rys)

def name_by_parameters(FLAGS):
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    is_aug = FLAGS.aug
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    network = FLAGS.network
    
    save_path = f'{FLAGS.weights_path}_B{bin_num}'
    if is_group>0:
        save_path += f'_G{is_group}_W{warm_up}'
    if is_cond==1:
        save_path += '_C'
    if is_aug==1:
        save_path += '_aug'
    
    if network==0:
        save_path += '_vgg'
    elif network==1:
        save_path += '_resnet'
    elif network==2:
        save_path += '_dense'

    train_config = save_path.split("weights/")[1]
    log_path = f'log/{train_config}'

    return save_path, log_path, train_config

if __name__=='__main__':
    main()
