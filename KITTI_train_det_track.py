from torch_lib.KITTI_Dataset import *
from torch_lib.Model_heading_bin import *
from library.ron_utils import *
from torch_lib.ClassAverages import *

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
parser.add_argument("--depth", "-DEP", type=int, default=0, help='if True, add depth loss')
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
    is_depth =FLAGS.depth
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
    W_angle = 0.1 #數值大0.05~0.2
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

    start = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        
        # DETECTION PART
        det_bin_loss, det_residual_loss, det_theta_loss = 0, 0, 0
        det_dim_loss, det_angle_loss, det_total_loss = 0, 0, 0
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
            theta_loss = bin_loss + residual_loss
            
            
            GT_alphas = labels_L['Alpha'].to(device)
            #dim_loss = F.mse_loss(dim, gt_dim, reduction='mean')  # org use mse_loss
            dim_loss = W_dim * F.l1_loss(dim_L, gt_dim, reduction='mean')  # 0613 added (monodle, monogup used) (compare L1 vs mse loss)
            #dim_loss = W_dim * L1_loss_alpha(dim_L, gt_dim, GT_alphas, device) # 0613 try elevate dim performance       
            #dim_loss = W_dim * F.mse_loss(dim_L, gt_dim, reduction='mean').to(device) # 0613 try elevate dim performance

            loss_detection = dim_loss + theta_loss
            #added loss
            if is_group > 0 and epoch > warm_up:
                # before 0814 group_alpha_loss
                #REG_alphas = compute_alpha(bin_L, residual_L, angle_per_class).to(device)
                #group_loss = group_loss_func(REG_alphas, GT_alphas)
                # 0815 group_ry_loss
                GT_rys = labels_L['Ry'].to(device)
                REG_rys = compute_ry(bin_L, residual_L, gt_theta_ray_L, angle_per_class).to(device)
                group_loss = W_group * group_loss_func(REG_rys, GT_rys).to(device)
                loss_detection += group_loss
            else:
                group_loss = torch.tensor(0.0).to(device)

            # 0801 added consist loss
            if type_!= 3: # baseline
                reg_ry_L = compute_ry(bin_L, residual_L, gt_theta_ray_L, angle_per_class)
                with torch.no_grad(): #0817 added
                    [residual_R, bin_R, dim_R] = model(batch_R)
                    reg_ry_R = compute_ry(bin_R, residual_R, gt_theta_ray_R, angle_per_class)
                    ry_angle_loss = W_angle * F.l1_loss(torch.cos(reg_ry_L), torch.cos(reg_ry_R), reduction='mean').to(device)
                #  dim_consist
                if type_==0:
                    ry_angle_loss = torch.tensor(0.0).to(device)
                # angle_consist
            else:
                ry_angle_loss = torch.tensor(0.0).to(device)

            loss_detection += ry_angle_loss
            
            loss_detection.backward()
            optimizer.step()

            det_bin_loss += bin_loss.item()*len(batch_L)
            det_residual_loss += residual_loss.item()*len(batch_L)
            det_theta_loss += theta_loss.item()*len(batch_L)
            det_dim_loss += dim_loss.item()*len(batch_L)
            #det_depth_loss += depth_loss.item()*len(batch_L) # my
            det_total_loss += loss_detection.item()*len(batch_L)
            if type_!= 3:
                det_angle_loss += ry_angle_loss.item()*len(batch_L) # my

        det_bin_loss/=len(dataset_train_all)
        det_residual_loss/=len(dataset_train_all)
        det_theta_loss/=len(dataset_train_all)
        det_dim_loss/=len(dataset_train_all)
        #det_depth_loss/=len(dataset_train_all)
        det_total_loss/=len(dataset_train_all)
        if type_!=3:
            det_angle_loss/=len(dataset_train_all)

        print('DET_count', len(dataset_train_all))
        print("--- DET epoch %s --- [loss: %.3f],[bin_loss:%.3f],[residual_loss:%.3f],[dim_loss:%.3f],[angle_loss:%.3f]" \
                %(epoch, det_total_loss, det_bin_loss, det_residual_loss, det_dim_loss, det_angle_loss))

        # TRACKING PART
        track_count, batch_count = 0, 0
        trk_bin_loss, trk_residual_loss, trk_theta_loss, trk_dim_loss = 0, 0, 0, 0
        trk_consist_loss, trk_angle_loss, trk_total_loss = 0, 0, 0
        lines, Frames, Track_id, reg_alphas = list(), list(), list(), list()
        frame, last_frame = 0, 0
        for folder in sorted(os.listdir('Kitti/training/image_02')):
            #print(folder)
            label = f'Kitti/training/label_02/{folder}.txt'
            track_labels = open(label).readlines()
            last_frame = 0, 0
            last_Track_id = None
            crops = list()
            for t_label in track_labels:
                line = t_label.split()
                frame = int(line[0])
                track_id = int(line[1])
                class_ = line[2]
                
                if class_.lower() not in cfg['class_list']:
                    continue
                truncation = float(line[3])
                occlusion = float(line[4])
                height = float(line[9]) - float(line[7]) + 1
                if not (height >= 40 and truncation <= 0.15 and occlusion <= 0) and not (height >= 25 and truncation <= 0.3 and occlusion <= 1):
                    continue
                # obj in the same frame_idx
                if frame == last_frame:
                    #print(t_label[len(line[0])+len(line[1])+2:])
                    lines.append(t_label[len(line[0])+len(line[1])+2:])
                    Frames.append(frame)
                    Track_id.append(track_id)
                # when it comes to new frame-obj 
                if frame != last_frame and len(lines) !=0:
                    optimizer.zero_grad()
                    
                    img = cv2.cvtColor(cv2.imread(f'Kitti/training/image_02/{folder}/{frame:06}.png'), cv2.COLOR_BGR2RGB)
                    objects = [Object3d(line) for line in lines]
                    for obj, f, id_ in zip(objects, Frames, Track_id):
                        if obj.level in cfg['diff_list']:
                            obj.set_track_info(f, id_)
                            crops.append(process(img[obj.box2d[1]:obj.box2d[3]+1, obj.box2d[0]:obj.box2d[2]+1]))
                    
                    crops = torch.stack(crops).to(device)
                    gt_labels = get_object_label(objects, bin_num)
                    gt_bin = gt_labels['bin'].to(device)
                    gt_residual = gt_labels['residual'].to(device)
                    gt_dim_delta = gt_labels['dim_delta'].to(device)
                    
                    [reg_residual, reg_bin, reg_dim_delta] = model(crops)
                    bin_loss = W_theta * F.cross_entropy(reg_bin, gt_bin, reduction='mean').to(device)
                    residual_loss = W_theta * compute_residual_loss(reg_residual, gt_bin, gt_residual, device)
                    theta_loss = bin_loss + residual_loss
                    dim_loss = W_dim * F.l1_loss(reg_dim_delta, gt_dim_delta, reduction='mean').to(device)

                    if type_ == 3:
                        consist_loss = torch.tensor(0.0).to(device)
                        angle_loss = torch.tensor(0.0).to(device)
                    else:
                        reg_alphas = compute_alpha(reg_bin, reg_residual, angle_per_class)  
                        # consist loss
                        if last_Track_id == None or len(last_Track_id)==0:
                            consist_loss = torch.tensor(0.0).to(device)
                            angle_loss = torch.tensor(0.0).to(device)
                        else:
                            now_id_list, last_id_list = id_compare(Track_id, last_Track_id)
                            if len(now_id_list) != 0:
                                #0719 added
                                now_dim_delta = reg_dim_delta[now_id_list]
                                last_dim_delta = last_dim_delta[last_id_list]
                                consist_loss = W_consist*F.l1_loss(now_dim_delta, last_dim_delta, reduction='sum').to(device)/len(Track_id)
                                #0721 added
                                now_alphas = reg_alphas[now_id_list]
                                last_alphas = compute_alpha(last_bin, last_residual, angle_per_class)[last_id_list]
                                angle_loss = W_angle * F.l1_loss(torch.cos(now_alphas), torch.cos(last_alphas), reduction='sum').to(device)/len(Track_id)
                                
                                # dim_consist
                                if type_==0:
                                    angle_loss = torch.tensor(0.0).to(device)
                                # alpha_consist
                                elif type_==1:
                                    consist_loss = torch.tensor(0.0).to(device)
                                # if type_==2 : both calculated

                            else:
                                consist_loss = torch.tensor(0.0).to(device)
                                angle_loss = torch.tensor(0.0).to(device)

                    loss_track = theta_loss + dim_loss + consist_loss + angle_loss
                    loss_track *= len(lines) / batch_size
                    loss_track.backward()

                    trk_bin_loss += bin_loss.item()*len(lines)
                    trk_residual_loss += residual_loss.item()*len(lines)
                    trk_theta_loss += theta_loss.item()*len(lines)
                    trk_dim_loss += dim_loss.item()*len(lines)
                    #trk_depth_loss += depth_loss.item()*len(lines) # my
                    trk_total_loss += loss_track.item()*batch_size
                    trk_consist_loss += consist_loss.item()*len(lines) # my
                    trk_angle_loss += angle_loss.item()*len(lines) # my

                    last_bin = torch.clone(reg_bin).detach()
                    last_residual = torch.clone(reg_residual).detach()
                    last_dim_delta = torch.clone(reg_dim_delta).detach()
                    
                    last_Track_id = Track_id
                    batch_count += len(lines)
                    track_count += len(lines)
                    # next frame info
                    lines = [t_label[len(line[0])+len(line[1])+2:]]
                    Frames = [frame]
                    Track_id = [track_id]
                    crops = list()
                    #print(now_id, last_id)
                    
                last_frame = frame
                if(batch_count//batch_size)==1:
                    #print(batch_count)
                    batch_count = 0 
                    optimizer.step()   

        trk_bin_loss/=track_count
        trk_residual_loss/=track_count
        trk_dim_loss/=track_count
        trk_theta_loss/=track_count
        #trk_depth_loss/=track_count
        trk_total_loss/=track_count
        trk_consist_loss/=track_count
        trk_angle_loss/=track_count

        print('Track count', track_count)
        print("--- Track epoch %s --- [loss: %.3f],[bin_loss:%.3f],[residual_loss:%.3f],[dim_loss:%.3f],[angle_loss:%.3f], [consist_loss:%.3f]," \
                %(epoch, trk_total_loss, trk_bin_loss, trk_residual_loss, trk_dim_loss, trk_angle_loss, trk_consist_loss))
        
        writer.add_scalars(f'{train_config}/bin_loss',  {'DET': det_bin_loss, 'TRK': trk_bin_loss}, epoch)
        writer.add_scalars(f'{train_config}/residual_loss', {'DET': det_residual_loss, 'TRK': trk_residual_loss}, epoch)
        writer.add_scalars(f'{train_config}/dim_loss', {'DET': det_dim_loss, 'TRK': trk_dim_loss}, epoch)
        writer.add_scalars(f'{train_config}/theta_loss', {'DET': det_theta_loss, 'TRK': trk_theta_loss}, epoch)
        writer.add_scalars(f'{train_config}/total_loss', {'DET': det_total_loss, 'TRK': trk_total_loss}, epoch)
        writer.add_scalars(f'{train_config}/consist_loss', {'DET': 0, 'TRK': trk_total_loss}, epoch)
        writer.add_scalars(f'{train_config}/ry_angle_loss', {'DET': det_angle_loss, 'TRK': trk_angle_loss}, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        #tensorboard --logdir=./{log_foler} --port 8123
            
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
                    'W_dim': W_dim,
                    'W_theta': W_theta,
                    'W_consist': W_consist,
                    'W_angle': W_angle,
                    'W_group': W_group,
                    }, name)
            print("====================")
            
    writer.close()
    print(f'Elapsed time:{(time.time()-start)//60}min')

#different from Elan get_object_label
def get_object_label(objects, bin_num=4):
    Kitti_averages = ClassAverages(average_file='Kitti_class_averages.txt')
    label = dict()
    Heading_class = list()
    Residual = list()
    Dim_delta = list()
    TrackID = list()
    for obj in objects:
        heading_class, residual = angle2class(obj.alpha, bin_num)
        dim_delta = np.array(obj.dim) - Kitti_averages.get_item(obj.cls_type)
        obj_id = obj.track_id
        Heading_class.append(heading_class)
        Residual.append(residual)
        Dim_delta.append(dim_delta)
        TrackID.append(obj_id)
    label['bin'] = torch.tensor(Heading_class)
    label['residual'] = torch.tensor(Residual)
    label['dim_delta'] = torch.tensor(np.array(Dim_delta))
    label['track_id'] = TrackID
    label['Group'] = list()
    label['Theta'] = list()
    return label

def id_compare(now_id, last_id):
    now_id_list = list()
    last_id_list = list()
    for idx, id_ in enumerate(now_id):
        try:
            find_idx = last_id.index(id_)
        except:
            find_idx = -1

        if find_idx != -1:
            now_id_list.append(idx)
            last_id_list.append(find_idx)
    return now_id_list, last_id_list

def name_by_parameters(FLAGS):
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    is_aug = FLAGS.aug
    is_depth = FLAGS.depth
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
