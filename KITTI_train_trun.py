from torch_lib.KITTI_Dataset import *
from torch_lib.Model_heading_bin_trun import *
from library.ron_utils import *
from library.Math_tensor import *

import torch
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
parser.add_argument("--latest-path", "-L_PATH", default='', help='continue training')

#training setting
parser.add_argument("--network", "-N", type=int, default=0, help='vgg/resnet/densenet')
parser.add_argument("--type", "-T", type=int, default=0, help='0:BL, 1:dim, 2:alpha, 3:both')
parser.add_argument("--device", "-D", type=int, default=0, help='select cuda index')
parser.add_argument("--epoch", "-E", type=int, default=50, help='epoch num')

#parser.add_argument("--batch-size", type=int, default=16, help='batch size')

# hyper-parameter (group | bin | cond)
parser.add_argument("--bin", "-B", type=int, default=4, help='heading bin num')
parser.add_argument("--group", "-G", type=int, help='if True, add stdGroupLoss')
parser.add_argument("--warm-up", "-W", type=int, default=10, help='warm up before adding group loss')
parser.add_argument("--cond", "-C", type=int, help='if True, 4-dim with theta_ray | boxH_2d ')
parser.add_argument("--aug", "-A", type=int, default=0, help='if True, flip dataset as augmentation')
parser.add_argument("--depth", "-DEP", type=int, default=0, help='if True, add depth loss')
parser.add_argument("--iou", "-IOU", type=int, default=0, help='if True, add iou loss')

# TO BE ADDED (Loss的weights比例alpha, w of groupLoss, LRscheduler: milestone, gamma )

def main():
    cfg = {'path':'Kitti/training',
            'class_list':['car'], 'diff_list': [1, 2], #0:DontCare, 1:Easy, 2:Moderate, 3:Hard, 4:Unknown
            'bins': 4, 'cond':False, 'group':False, 'network':0}
    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    print('DEVICE:', device)
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    is_aug = FLAGS.aug
    is_depth =FLAGS.depth
    is_iou = FLAGS.iou
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    type_ = FLAGS.type
    epochs = FLAGS.epoch
    network = FLAGS.network
    batch_size = 16 #64 worse than 8
    weight_dict = {'dim':1, 'theta':1, 'group': 1, 'C_dim':1, 'C_angle':0.2, 'depth':0.05, 'iou':0.2}
    # make weights folder
    cfg['bins'] = bin_num
    cfg['cond'] = is_cond
    cfg['group'] = is_group
    cfg['network'] = network
    start_epoch = 0
    best_value = 100 # for record best epoch



    # continue training
    if len(FLAGS.latest_path) > 0:
        print('-------------CONTINUE TRAINING AND LOAD CHECKPOINT----------------')
        checkpoint = torch.load(FLAGS.latest_path, map_location=device) #if training on 2 GPU, mapping on the same device
        start_epoch = checkpoint['epoch']
        cfg = checkpoint['cfg']
        if 'weight_dict' in checkpoint.keys():
            weight_dict = checkpoint['weight_dict']
        network = cfg['network']
        weight_dict = checkpoint['weight_dict']
        #best_value = checkpoint['best_value']
        #best_epoch = checkpoint['best_epoch']
    
    weights_folder = os.path.join('weights', FLAGS.weights_path.split('/')[1])
    print('weights_folder:', weights_folder)
    os.makedirs(weights_folder, exist_ok=True)
    save_path, log_path, train_config = name_by_parameters(FLAGS)
    print(f'SAVE PATH:{save_path}, LOG PATH:{log_path}, config:{train_config}')
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    if is_group == 1:
        group_loss_func = cos_std_loss
    elif is_group == 2:
        group_loss_func = sin_sin_std_loss
    elif is_group == 3:
        group_loss_func = compare_abs_best_loss
    if is_group > 0:
        print('Group loss:', str(group_loss_func))


    # model
    print("Loading all detected objects in dataset...")
    start = time.time()
    process = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize([224,224], transforms.InterpolationMode.BICUBIC), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = KITTI_Dataset(cfg, process, split='train')
    dataset_valid = KITTI_Dataset(cfg, process, split='val')
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}
    angle_per_class=2*np.pi/float(bin_num)
    if is_aug:
        dataset_train_flip = KITTI_Dataset(cfg, process, split='train', is_flip=True)
        dataset_train_all = data.ConcatDataset([dataset_train, dataset_train_flip])
    else:
        dataset_train_all = dataset_train
    train_loader = data.DataLoader(dataset_train_all, **params)
    valid_loader = data.DataLoader(dataset_valid, **params)
    print(f'LOAD time:{(time.time()-start)//60}min')

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
        #model.features[0] = torch.nn.Conv2d(4, 64, (3,3), (1,1), (1,1))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    # milestones:調整lr的epoch數，gamma:decay factor (https://hackmd.io/@Hong-Jia/H1hmbNr1d)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(10, epochs, 20)], gamma=0.5)
    #dim_loss_func = nn.MSELoss().to(device) #org function

    if len(FLAGS.latest_path) > 0:
        print('-------------CONTINUE TRAINING AND LOAD STATE DICT----------------')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print('total_batches', len(train_loader))
    start = time.time()
    for epoch in range(start_epoch+1, epochs+1):
        # ---------------------------------------- TRAINING PART ---------------------------------------- #
        model.train()
        train_loss_dict = init_loss_dict()
        before_par = [par.clone() for par in list(model.parameters())]
        for batch_L, labels_L, batch_R, labels_R in train_loader:
            optimizer.zero_grad()
            batch_L=batch_L.to(device)
            batch_R=batch_R.to(device)
            gt_residual = labels_L['Heading_res'].to(device)
            gt_bin = labels_L['Heading_bin'].to(device)#這個角度在哪個class上
            gt_dim = labels_L['Dim_delta'].to(device)
            gt_theta_ray_L = labels_L['Theta_ray'].to(device)
            gt_theta_ray_R = labels_R['Theta_ray'].to(device)
            gt_depths = labels_L['Depth'].to(device)
            gt_alphas = labels_L['Alpha'].to(device)
            gt_rys = labels_L['Ry'].to(device)
            gt_img_W = labels_L['img_W'] #depth_loss
            gt_box2d = labels_L['Box2d'] #depth_loss
            gt_calib = labels_L['Calib'].numpy() #depth_loss, iou_loss
            gt_trun = labels_L['Truncation'] #depth_loss
            
            [residual_L, bin_L, dim_L, trun_L] = model(batch_L)

            bin_loss = weight_dict['theta'] * F.cross_entropy(bin_L, gt_bin, reduction='mean').to(device)
            residual_loss = weight_dict['theta'] * compute_residual_loss(residual_L, gt_bin, gt_residual, device)
            #dim_loss = F.mse_loss(dim, gt_dim, reduction='mean')  # org use mse_loss
            dim_loss = weight_dict['dim'] * F.l1_loss(dim_L, gt_dim, reduction='mean')  # 0613 added (monodle, monogup used) (compare L1 vs mse loss)
            #dim_loss = W_dim * L1_loss_alpha(dim_L, gt_dim, GT_alphas, device) # 0613 try elevate dim performance       
            #dim_loss = W_dim * F.mse_loss(dim_L, gt_dim, reduction='mean').to(device) # 0613 try elevate dim performance

            #trun_loss = 1 * F.l1_loss(trun_L, gt_trun.view(-1,1).to(device), reduction='mean')
            trun_loss = 1 * F.mse_loss(trun_L, gt_trun.view(-1,1).to(device), reduction='mean')

            # consistency loss
            if type_ > 0:
                with torch.no_grad(): #0817 added
                    [residual_R, bin_R, dim_R] = model(batch_R)

            if type_ == 1 or type_ == 3: # D, DA
                C_dim_loss = weight_dict['C_dim'] * F.l1_loss(dim_L, dim_R, reduction='mean').to(device)
            else:
                C_dim_loss = torch.tensor(0.0).to(device)

            if type_ == 2 or type_ == 3: # A, DA
                reg_ry_L = compute_angle_by_bin_residual(bin_L, residual_L, angle_per_class, gt_theta_ray_L) #grad會斷掉 連不到bin_L, residual
                reg_ry_R = compute_angle_by_bin_residual(bin_R, residual_R, angle_per_class, gt_theta_ray_R)
                C_angle_loss = weight_dict['C_angle'] * F.l1_loss(torch.cos(reg_ry_L), torch.cos(reg_ry_R), reduction='mean').to(device)
            else:
                C_angle_loss = torch.tensor(0.0).to(device)

            # group_loss
            if is_group > 0 :
                # before 0814 group_alpha_loss
                #REG_alphas = compute_alpha(bin_L, residual_L, angle_per_class).to(device)
                #group_loss = group_loss_func(REG_alphas, GT_alphas)
                # 1021 group_ry_loss 
                #cos_std_loss | sin_sin_std_loss | compare_cos TODO check performance
                reg_rys = compute_angle_by_bin_residual(bin_L, residual_L, angle_per_class, gt_theta_ray_L).to(device)
                group_loss = weight_dict['group'] * compute_group_loss(reg_rys, gt_rys, loss_func=group_loss_func)
            else:
                group_loss = torch.tensor(0.0).to(device)

            reg_alphas = compute_angle_by_bin_residual(bin_L, residual_L, angle_per_class)
            reg_dims = torch.tensor(dataset_train.get_cls_dim_avg('car')).to(device) + dim_L
            #gt_dims = torch.tensor(dataset_train.get_cls_dim_avg('car')).to(device) + gt_dim
            if is_depth > 0:
                obj_W, obj_L = reg_dims[:,1], reg_dims[:,2]
                if is_depth==1:
                    depth_alphas = reg_alphas #dep
                elif is_depth==2:
                    depth_alphas = gt_alphas #depA            
                calc_depths = calc_depth_with_alpha_theta_tensor(gt_img_W, gt_box2d, gt_calib, obj_W, obj_L, depth_alphas, gt_trun, device)
                depth_loss = weight_dict['depth'] * F.l1_loss(calc_depths, gt_depths, reduction='mean')
            else:
                depth_loss = torch.tensor(0.0).to(device)

            if is_iou > 0:
                if is_iou == 1:
                    iou_alphas = reg_alphas
                elif is_iou == 2:
                    iou_alphas =  torch.tensor(gt_alphas)
                #iou_alphas = reg_alphas if is_iou==1 else torch.tensor(gt_alphas)
                iou_loss = weight_dict['iou'] * calc_IoU_loss_tensor(gt_box2d, gt_theta_ray_L, reg_dims.cpu(), iou_alphas, gt_calib, device) #iou
            else:
                iou_loss = torch.tensor(0.0).to(device)

            total_loss = dim_loss + bin_loss + residual_loss + group_loss \
                        + C_dim_loss + C_angle_loss + depth_loss + iou_loss + trun_loss
            
            train_loss_dict = loss_dict_add(train_loss_dict, batch_L.shape[0], bin=bin_loss.item(), residual=residual_loss.item(), \
                                            dim=dim_loss.item(), total=total_loss.item(), group=group_loss.item(), C_dim=C_dim_loss.item(), \
                                            C_angle=C_angle_loss.item(), depth=depth_loss.item(), iou=iou_loss.item(), \
                                            trun=trun_loss.item()) #.item()

            total_loss.backward()
            optimizer.step()
        
        train_loss_dict = calc_avg_loss(train_loss_dict, len(dataset_train))
        print_epoch_loss(train_loss_dict, epoch, type='Train')

        after_par = [par.clone() for par in list(model.parameters())]
        is_equal = [torch.equal(a.data, b.data) for a, b in zip(before_par, after_par)]
        print(f'IS EQUAL? {sum(is_equal) == len(before_par)} | Equal: {sum(is_equal)} | Not Equal: {len(before_par) - sum(is_equal)}', )
        #print('model grad:', list(model.parameters())[-13].grad) # -1~-12 not update with some loss_func
        if sum(is_equal) == len(before_par):
            print(f'{save_path} | SOMEWHERE WRONG!')
            exit()

        # ---------------------------------------- EVALUATION PART ---------------------------------------- #
        model.eval()
        val_loss_dict = init_loss_dict()
        VAL_alpha_list = list()
        VAL_dim_list = list()
        VAL_reg_alpha_list = list()
        VAL_reg_dim_list = list()
        VAL_reg_trun_list = list()
        with torch.no_grad():
            for batch_L, labels_L, batch_R, labels_R in valid_loader:
                batch_L=batch_L.to(device)
                batch_R=batch_R.to(device)
                gt_residual = labels_L['Heading_res'].to(device)
                gt_bin = labels_L['Heading_bin'].to(device)#這個角度在哪個class上
                gt_dim = labels_L['Dim_delta'].to(device)
                gt_theta_ray_L = labels_L['Theta_ray'].to(device)
                gt_theta_ray_R = labels_R['Theta_ray'].to(device)
                gt_depths = labels_L['Depth'] #val compute on cpu
                gt_alphas = labels_L['Alpha'].to(device)
                gt_rys = labels_L['Ry'].to(device)
                gt_img_W = labels_L['img_W'].numpy() #depth_loss, tensor->numpy
                gt_box2d = labels_L['Box2d'].numpy() #depth_loss
                gt_calib = labels_L['Calib'].numpy() #depth_loss, iou_loss
                gt_trun = labels_L['Truncation'] #depth_loss
                
                [residual_L, bin_L, dim_L, trun_L] = model(batch_L)
                # ORG loss
                bin_loss = weight_dict['theta'] * F.cross_entropy(bin_L, gt_bin, reduction='mean').to(device)
                residual_loss = weight_dict['theta'] * compute_residual_loss(residual_L, gt_bin, gt_residual, device)
                dim_loss = weight_dict['dim'] * F.l1_loss(dim_L, gt_dim, reduction='mean')
                #trun_loss = 1 * F.l1_loss(trun_L, gt_trun.view(-1,1).to(device), reduction='mean')
                trun_loss = 1 * F.mse_loss(trun_L, gt_trun.view(-1,1).to(device), reduction='mean')
                # consistency dimension loss
                if type_ > 0:
                    [residual_R, bin_R, dim_R] = model(batch_R)
                if type_ == 1 or type_ == 3: # D, DA
                    C_dim_loss = weight_dict['C_dim'] * F.l1_loss(dim_L, dim_R, reduction='mean').to(device)
                else:
                    C_dim_loss = torch.tensor(0.0).to(device)
                # consistency Ry angle loss
                if type_ == 2 or type_ == 3: # A, DA
                    reg_ry_L = compute_angle_by_bin_residual(bin_L, residual_L, angle_per_class, gt_theta_ray_L) #grad會斷掉 連不到bin_L, residual
                    reg_ry_R = compute_angle_by_bin_residual(bin_R, residual_R, angle_per_class, gt_theta_ray_R)
                    C_angle_loss = weight_dict['C_angle'] * F.l1_loss(torch.cos(reg_ry_L), torch.cos(reg_ry_R), reduction='mean').to(device)
                else:
                    C_angle_loss = torch.tensor(0.0).to(device)
                # group_loss
                if is_group > 0 :
                    reg_rys = compute_angle_by_bin_residual(bin_L, residual_L, angle_per_class, gt_theta_ray_L).to(device)
                    group_loss = weight_dict['group'] * compute_group_loss(reg_rys, gt_rys, loss_func=group_loss_func)
                else:
                    group_loss = torch.tensor(0.0).to(device)

                reg_alphas = compute_angle_by_bin_residual(bin_L, residual_L, angle_per_class)
                reg_dims = torch.tensor(dataset_train.get_cls_dim_avg('car')).to(device) + dim_L
                #gt_dims = torch.tensor(dataset_train.get_cls_dim_avg('car')).to(device) + gt_dim
                #compute on GPU
                if is_depth > 0:
                    obj_W, obj_L = reg_dims[:,1], reg_dims[:,2]
                    if is_depth==1:
                        depth_alphas = reg_alphas #dep
                    elif is_depth==2:
                        depth_alphas = gt_alphas #depA            
                    calc_depths = calc_depth_with_alpha_theta_tensor(gt_img_W, gt_box2d, gt_calib, obj_W, obj_L, depth_alphas, gt_trun, device)
                    depth_loss = weight_dict['depth'] * F.l1_loss(calc_depths, gt_depths, reduction='mean')
                else:
                    depth_loss = torch.tensor(0.0).to(device)

                #compute on cpu
                if is_iou > 0:
                    if is_iou == 1:
                        iou_alphas = reg_alphas.cpu().detach().numpy()
                    elif is_iou == 2:
                        iou_alphas = gt_alphas.cpu().detach().numpy()
                    iou_loss = weight_dict['iou'] * calc_IoU_loss(gt_box2d, gt_theta_ray_L.detach().cpu().numpy(), \
                                                                        reg_dims, iou_alphas, gt_calib).to(device)
                else:
                    iou_loss = torch.tensor(0.0).to(device)

                total_loss = dim_loss + bin_loss + residual_loss + group_loss \
                            + C_dim_loss + C_angle_loss + depth_loss + iou_loss + trun_loss
                
                
                val_loss_dict = loss_dict_add(val_loss_dict, batch_L.shape[0], bin=bin_loss.item(), residual=residual_loss.item(), \
                                            dim=dim_loss.item(), total=total_loss.item(), group=group_loss.item(), C_dim=C_dim_loss.item(), \
                                            C_angle=C_angle_loss.item(), depth=depth_loss.item(), iou=iou_loss.item(), \
                                            trun=trun_loss.item()) #.item()
            
                VAL_reg_alpha_list += reg_alphas.cpu().tolist()
                VAL_alpha_list += gt_alphas.cpu().tolist()
                VAL_reg_dim_list += dim_L.cpu().tolist()
                VAL_dim_list += gt_dim.cpu().tolist()

            val_loss_dict = calc_avg_loss(val_loss_dict, len(dataset_valid))
            print_epoch_loss(val_loss_dict, epoch, type='Valid')

        eval_angle_diff = angle_criterion(VAL_reg_alpha_list, VAL_alpha_list)
        VAL_dim_list = np.array(VAL_dim_list)
        VAL_reg_dim_list = np.array(VAL_reg_dim_list)
        eval_dim_diff =  np.mean(abs(VAL_dim_list-VAL_reg_dim_list), axis=0)
        print(f'[Alpha diff]: {eval_angle_diff:.4f}') #close to 0 is better
        print(f'[DIM diff] H:{eval_dim_diff[0]:.4f}, W:{eval_dim_diff[1]:.4f}, L:{eval_dim_diff[2]:.4f}')

        # Write tmp performance
        writer.add_scalar(f'{train_config}/alpha_diff', eval_angle_diff, epoch)
        writer.add_scalar(f'{train_config}/H_diff', eval_dim_diff[0], epoch)
        writer.add_scalar(f'{train_config}/W_diff', eval_dim_diff[1], epoch)
        writer.add_scalar(f'{train_config}/L_diff', eval_dim_diff[2], epoch)
        #write LOSS
        for key in train_loss_dict.keys():
            writer.add_scalars(f'{train_config}/{key}_loss', {'Train': train_loss_dict[key], 'Valid': val_loss_dict[key]}, epoch)

        overall_value = eval_angle_diff*3 + np.sum(eval_dim_diff)
        if overall_value < best_value:
            best_value = overall_value
            best_epoch = epoch
            best_parameters = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cfg': cfg,
                'weight_dict': weight_dict,
                'best_value': best_value,
                'best_epoch': best_epoch
            }
            name = f'{save_path}_best.pkl'
            print("====================")
            print (f"Best weights save @ {epoch}")
            torch.save(best_parameters, name)
            print("====================")
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        #tensorboard --logdir=./{log_foler} --port 8123
        # 1007 added record model weight (check updated)
        #plot_params_hist(writer, model, epoch)
            
        if epoch % epochs == 0 or epoch % 10 == 0:
            name = f'{save_path}_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cfg': cfg,
                    'weight_dict': weight_dict,
                    'best_value': best_value,
                    'best_epoch': best_epoch
                    }, name)
            print("====================")
        print(f'Elapsed time:{(time.time()-start)//60}min')

    writer.close()

    # remove not best pkl
    for epoch in range(0, epochs, 10):
        if epoch < best_epoch:
            tmp_pkl = f'{save_path}_{epoch}.pkl'
            if os.path.exists(tmp_pkl):
                print(f'REMOVE {tmp_pkl}')
                os.remove(tmp_pkl)
    
# record model weights
def plot_params_hist(writer, model, epoch):
    for name, param in model.named_parameters():
        #writer.add_histogram(tag=name)
        if 'dimension' in name or 'confidence' in name or 'orientation' in name:
            if 'weight' in name:
                #print(name, param.shape)
                writer.add_histogram(tag=name, values=param.data.clone().cpu().numpy(), global_step=epoch)

def name_by_parameters(FLAGS):
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    is_aug = FLAGS.aug
    is_depth = FLAGS.depth
    is_iou = FLAGS.iou
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    network = FLAGS.network
    
    save_path = f'{FLAGS.weights_path}_B{bin_num}'
    if is_group>0:
        save_path += f'_G{is_group}_W{warm_up}'
    if is_cond==1:
        save_path += '_C'
    if is_depth==1:
        save_path += '_dep'
    elif is_depth==2:
        save_path += '_depA'
    if is_iou==1:
        save_path += '_iou'
    if is_iou==2:
        save_path += '_iouA'
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
