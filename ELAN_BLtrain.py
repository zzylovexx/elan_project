from torch_lib.ELAN_Dataset_new import *
from torch_lib.Model_heading_bin import *
from library.ron_utils import *

import torch
import torch.nn.functional as F
from torchvision.models import vgg
from torch.utils import data
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
import os, time
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
    
    cfg = {'path':'Elan_3d_box',
        'class_list':['car'], 'diff_list': [1, 2], #0:DontCare, 1:Easy, 2:Moderate, 3:Hard, 4:Unknown
        'bins': 4, 'cond':False, 'group':False, 'network':0}

    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    is_aug = FLAGS.aug
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    normalize_type = FLAGS.normal
    W_theta = 1
    W_dim = 0.6 #alpha = 0.6
    W_iou = 0.2
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
    

    process = transforms.Compose([transforms.ToTensor(), 
                            transforms.Resize([224,224], transforms.InterpolationMode.BICUBIC), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # model
    #dataset_train_flip = KITTI_Dataset(cfg, process, split='train', is_flip=True)
    dataset_train = ELAN_Dataset(cfg, process, is_flip=False, img_extension='png')
    if is_aug:
        dataset_train_flip = ELAN_Dataset(cfg, process, is_flip=True, img_extension='png')
        dataset_train_all = data.ConcatDataset([dataset_train, dataset_train_flip])
    else:
        dataset_train_all = dataset_train
    print(f"Load all training files in ELAN dataset:{len(dataset_train)}")
    
    cfg_valid = {'path':'Elan_3d_box_230808', 'class_list':['car'], 'diff_list': [1, 2], 'bins': 4, 'cond':False, 'group':False, 'network':0}
    dataset_valid = ELAN_Dataset(cfg_valid, process, is_flip=False, img_extension='png')
    print(f"Load all validation files in ELAN dataset:{len(dataset_valid)}")

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    train_loader = data.DataLoader(dataset_train_all, **params)
    valid_loader = data.DataLoader(dataset_valid, **params)

    my_vgg = vgg.vgg19_bn(weights='DEFAULT')
    model = vgg_Model(features=my_vgg.features, bins=bin_num).to(device)    
    if is_cond:
        print("< 4-dim input, Theta_ray as Condition >")
        model.features[0].in_channels = 4
        #model.features[0] = torch.nn.Conv2d(4, 64, (3,3), (1,1), (1,1))

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
    
    start = time.time()
    for epoch in range(1, epochs+1):
        avg_bin_loss, avg_residual_loss, avg_theta_loss = 0, 0, 0
        avg_dim_loss, avg_consist_loss, avg_angle_loss, avg_depth_loss = 0,0,0,0
        avg_total_loss, avg_iou_loss = 0, 0
        model.train()
        for batch, labels in train_loader:
            optimizer.zero_grad()

            gt_residual = labels['Heading_res'].float().to(device)
            gt_bin = labels['Heading_bin'].long().to(device)#這個角度在哪個class上
            gt_dim = labels['Dim_delta'].float().to(device)
            gt_theta_ray = labels['Theta_ray'].numpy()
            gt_depth = labels['Depth'].float().to(device)
            gt_img_W = labels['img_W']
            gt_box2d = labels['Box2d'].numpy()
            gt_calib = labels['Calib'].numpy()
            gt_class = labels['Class']

            batch=batch.float().to(device)
            [residual, bin, dim] = model(batch)

            bin_loss = W_theta * F.cross_entropy(bin, gt_bin,reduction='mean').to(device)
            residual_loss = W_theta * compute_residual_loss(residual, gt_bin, gt_residual, device)
            theta_loss = bin_loss + residual_loss
            # performance alpha_l1 > l1 > mse
            #dim_loss = F.mse_loss(dim, truth_dim, reduction='mean')  # org use mse_loss
            dim_loss = W_dim * F.l1_loss(dim, gt_dim, reduction='mean')  # 0613 added (monodle, monogup used) (compare L1 vs mse loss)
            #dim_loss = L1_loss_alpha(dim, truth_dim, GT_alpha, device) # 0613 try elevate dim performance       
            
            #0919added IOU
            reg_alphas = compute_alpha(bin, residual, angle_per_class)
            reg_dims = dataset_train.get_cls_dim_avg('car') + dim.cpu().detach().numpy()
            #TODO IOU aug, but DIOU, CIOU is close to IOU=GIOU value....
            iou_loss = W_iou * calc_IoU_loss(gt_box2d, gt_theta_ray, reg_dims, reg_alphas, gt_calib).to(device)

            #added loss
            if is_group and epoch > warm_up:
                gt_Theta = labels['Theta'].float().to(device)
                #truth_Ry = local_labels['Ry'].float().to(device)
                gt_group = labels['Group'].float().to(device)
                bin_argmax = torch.max(bin, dim=1)[1]
                pred_alpha = angle_per_class*bin_argmax + residual[torch.arange(len(residual)), bin_argmax]
                #group_loss = group_loss_func(pred_alpha, truth_Theta, truth_group, device)
                #(orient_batch, orientGT_batch, confGT_batch, group_batch, device)
                group_loss = group_loss_func(pred_alpha, gt_Theta, gt_group, device)
            else:
                group_loss = torch.tensor(0.0).to(device)
            
            #loss = alpha * dim_loss + bin_loss + residual_loss + W_group * group_loss # W_group=0.3 before0724
            loss = dim_loss + theta_loss + iou_loss
            
            loss.backward()
            optimizer.step()

            avg_bin_loss += bin_loss.item()*len(batch)
            avg_residual_loss += residual_loss.item()*len(batch)
            avg_theta_loss += theta_loss.item()*len(batch)
            avg_dim_loss += dim_loss.item()*len(batch)
            #avg_depth_loss += depth_loss.item()*len(batch) # my
            avg_total_loss += loss.item()*len(batch)
            avg_iou_loss += iou_loss.item()*len(batch)

        avg_bin_loss/=len(dataset_train_all)
        avg_residual_loss/=len(dataset_train_all)
        avg_theta_loss/=len(dataset_train_all)
        avg_dim_loss/=len(dataset_train_all)
        #avg_depth_loss/=len(dataset_train_all)
        avg_total_loss/=len(dataset_train_all)
        avg_iou_loss/=len(dataset_train_all)

        # eval part
        model.eval()
        with torch.no_grad():
            eval_bin_loss, eval_residual_loss, eval_theta_loss = 0,0,0
            eval_dim_loss, eval_consist_loss, eval_angle_loss, eval_depth_loss = 0,0,0,0
            eval_total_loss, eval_iou_loss = 0, 0
            GT_alpha_list = list()
            REG_alpha_list = list()
            GT_dim_list = list()
            REG_dim_list = list()
            for batch, labels in valid_loader:
                gt_residual = labels['Heading_res'].float().to(device)
                gt_bin = labels['Heading_bin'].long().to(device)#這個角度在哪個class上
                gt_dim = labels['Dim_delta'].float().to(device)
                gt_theta_ray = labels['Theta_ray'].numpy()
                gt_depth = labels['Depth'].float().to(device)
                gt_img_W = labels['img_W']
                gt_box2d = labels['Box2d'].numpy()
                gt_calib = labels['Calib'].numpy()
                gt_class = labels['Class']

                batch = batch.float().to(device)
                [residual, bin, dim] = model(batch)

                bin_loss = W_theta * F.cross_entropy(bin, gt_bin, reduction='mean').to(device)
                residual_loss = W_theta * compute_residual_loss(residual, gt_bin, gt_residual, device)
                theta_loss = bin_loss + residual_loss

                
                dim_loss = W_dim * F.l1_loss(dim, gt_dim, reduction='mean')
                val_alphas = compute_alpha(bin, residual, angle_per_class)
                val_dims = dataset_train.get_cls_dim_avg('car') + dim.cpu().detach().numpy()
                iou_loss = W_iou * calc_IoU_loss(gt_box2d, gt_theta_ray, val_dims, val_alphas, gt_calib).to(device)

                # calc GT_alpha,return list type
                #GT_alphas = labels['Alpha']
                REG_alpha_list += val_alphas.tolist()
                GT_alpha_list += labels['Alpha'].tolist()
                REG_dim_list += dim.cpu().tolist()
                GT_dim_list += gt_dim.cpu().tolist()

                # sum loss
                eval_bin_loss += bin_loss.item()*len(batch)
                eval_residual_loss += residual_loss.item()*len(batch)
                eval_theta_loss += theta_loss.item()*len(batch)
                eval_dim_loss += dim_loss.item()*len(batch)
                #eval_depth_loss += depth_loss.item()*len(batch) # my
                eval_total_loss += loss.item()*len(batch)
                eval_iou_loss += iou_loss.item()*len(batch)

            eval_bin_loss/=len(dataset_valid)
            eval_residual_loss/=len(dataset_valid)
            eval_theta_loss/=len(dataset_valid)
            eval_dim_loss/=len(dataset_valid)
            eval_depth_loss/=len(dataset_valid)
            eval_total_loss/=len(dataset_valid)
            eval_iou_loss/=len(dataset_valid)
            
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
        writer.add_scalars(f'{train_config}/theta_loss', {'Train': avg_theta_loss, 'Valid': eval_theta_loss}, epoch)
        writer.add_scalars(f'{train_config}/total_loss', {'Train': avg_total_loss, 'Valid': eval_total_loss}, epoch)
        writer.add_scalars(f'{train_config}/iou_loss', {'Train': avg_iou_loss, 'Valid': eval_iou_loss}, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        # MobaXterm https://zhuanlan.zhihu.com/p/138811263
        #tensorboard --logdir=./{log_foler} --port 8123

        #scheduler.step()
        if epoch % epochs == 0 or epoch % 50 == 0:
            name = save_path + f'_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cfg': cfg,
                    'normal': normalize_type,
                    'W_dim': W_dim,
                    'W_theta': W_theta,
                    'W_iou': W_iou,
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
