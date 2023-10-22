from torchvision.models import vgg, resnet, densenet
from torch_lib.Model_heading_bin import *
from torchvision import transforms
from torch_lib.KITTI_Dataset import *
from torch_lib.ClassAverages import *
from KITTI_EVAL import ron_evaluation
from library.ron_utils import *
import os, cv2, time, sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", '-D', type=int, default=0, help='select cuda index')
# path setting
parser.add_argument("--weights-path", '-W_PATH', required=True, help='weighs path')
parser.add_argument("--result-path", '-R_PATH', required=True, help='path (folder name) of the generated pred-labels')
parser.add_argument("--network", "-N", type=int, default=0, help='vgg/resnet/densenet')

def main():
    
    #1022 added
    os.makedirs('KITTI_labels', exist_ok=True)
    
    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_path
    eval_root = FLAGS.result_path
    save_root = os.path.join('KITTI_labels', eval_root)
    network = FLAGS.network
    os.makedirs(save_root, exist_ok=True)

    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    checkpoint = torch.load(weights_path, map_location=device) #if training on 2 GPU, mapping on the same device
    cfg = checkpoint['cfg']
    bin_num = cfg['bins'] 
    is_cond = cfg['cond']
    diff_list = cfg['diff_list']
    cls_list = cfg['class_list']
    
    angle_per_class = 2*np.pi/float(bin_num)

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

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # for img processing
    process = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize([224,224], transforms.InterpolationMode.BICUBIC), 
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    dataset_train = KITTI_Dataset(cfg, process, split='train') #TODO change to load txt files for avg_dims
    Kitti_averages = ClassAverages(average_file='Kitti_class_averages.txt')
    img2_path = 'Kitti/training/image_2'
    label2_path = 'Kitti/training/label_2'
    calib_path = 'Kitti/training/calib'
    trainval_ids = [x.strip() for x in open('Kitti/ImageSets/trainval.txt').readlines()]
    train_ids = [x.strip() for x in open('Kitti/ImageSets/train.txt').readlines()]
    val_ids = [x.strip() for x in open('Kitti/ImageSets/val.txt').readlines()]

    train_GT_depth = list()
    train_CALC_depth = list()
    val_GT_depth = list()
    val_CALC_depth = list()
    with torch.no_grad():
        for id_ in trainval_ids:
            label2_txt = os.path.join(label2_path, f'{id_}.txt')
            cam_to_img = FrameCalibrationData(os.path.join(calib_path, f'{id_}.txt'))
            img2 = cv2.cvtColor(cv2.imread(os.path.join(img2_path, f'{id_}.png')), cv2.COLOR_BGR2RGB)
            objects = [Object3d(line) for line in open(label2_txt).readlines()]
            inputs = list()
            REG_objects = list()
            for obj in objects:
                if obj.cls_type in cls_list and obj.level in diff_list:
                    obj.set_crop(img2, cam_to_img, 'left')
                    inputs.append(process(obj.crop))
                    REG_objects.append(obj)
            reg_labels = ''
            if len(inputs)!=0:
                inputs = torch.stack(inputs).to(device)
                # model regress part
                [residual, bin_, dim] = model(inputs)
                bin_argmax = torch.max(bin_, dim=1)[1]
                orient_residual = residual[torch.arange(len(residual)), bin_argmax].detach()
                REG_alphas = angle_per_class*bin_argmax + orient_residual #mapping bin_class and residual to get alpha
            
                for i in range(len(inputs)):
                    obj = REG_objects[i]
                    reg_alpha = angle_correction(REG_alphas[i].detach().item())
                    avg_dim = np.array(dataset_train.get_cls_dim_avg(obj.cls_type))
                    reg_dim = avg_dim + dim[i].cpu().detach().numpy()
                    reg_pos, _ = calc_location(reg_dim, cam_to_img.p2, obj.box2d.reshape((2,2)), reg_alpha, obj.theta_ray)
                    reg_pos[1] += reg_dim[0]/2 #reg_pos is 3d center, + H/2 to be the same standard as gt label
                    reg_labels += obj.REG_result_to_kitti_format_label(alpha=reg_alpha, dim=reg_dim, pos=reg_pos) + '\n'
                    # [TRAIN]
                    if id_ in train_ids:
                        train_GT_depth.append(obj.pos[2])
                        train_CALC_depth.append(calc_depth_with_alpha_theta(img2.shape[1], obj.box2d, cam_to_img.p2, reg_dim[1], reg_dim[2], reg_alpha))
                    # [VAL] compare GT_depth and CALC_depth
                    if id_ in val_ids:
                        val_GT_depth.append(obj.pos[2])
                        val_CALC_depth.append(calc_depth_with_alpha_theta(img2.shape[1], obj.box2d, cam_to_img.p2, reg_dim[1], reg_dim[2], reg_alpha))
            
            with open(os.path.join(save_root, f'{id_}.txt'), 'w') as f:
                f.writelines(reg_labels)
    
    # eval part
    train_GT_depth = np.array(train_GT_depth)
    train_CALC_depth = np.array(train_CALC_depth)
    val_GT_depth = np.array(val_GT_depth)
    val_CALC_depth = np.array(val_CALC_depth)
    #write as file as well
    org_stdout = sys.stdout
    os.makedirs(f'KITTI_eval/{eval_root.split("/")[0]}', exist_ok=True)
    f = open(f'KITTI_eval/{eval_root}.txt', 'w')
    sys.stdout = f
    print_info(checkpoint, cfg)
    ron_evaluation(val_ids, diff_list, cls_list, save_root)
    print('[MY CALC Depth error]')
    box_depth_error_calculation(val_GT_depth, val_CALC_depth, 5)
    print('=============[Train EVAL]===============')
    ron_evaluation(train_ids, diff_list, cls_list, save_root)
    print('[MY CALC Depth error]')
    box_depth_error_calculation(train_GT_depth, train_CALC_depth, 5)
    sys.stdout = org_stdout
    f.close()
    print(f'save in KITTI_eval/{eval_root}.txt')
    
def print_info(checkpoint, cfg):
    class_list = cfg['class_list']
    diff_list = cfg['diff_list']
    group = cfg['group']
    cond = cfg['cond']
    print('Class:', class_list, end=', ')
    print('Diff:', diff_list, end=', ')
    print(f'Group:{group}, cond:{cond}')
    #1020 updated
    if 'weight_dict' in checkpoint.keys():
        print(f'Best@{checkpoint["best_epoch"]}:{checkpoint["best_value"]:.4f}  [Weights] ', end='')
        for key in checkpoint['weight_dict'].keys():
            print(f'{key}:{checkpoint["weight_dict"][key]:.2f}', end=', ')
        print()
    else:
        W_consist = checkpoint['W_consist']
        W_ry = checkpoint['W_ry']
        W_group = checkpoint['W_group']

        print(f'[Weights] W_consist:{W_consist:.2f}, W_ry:{W_ry:.2f}, W_group:{W_group:.2f}', end='')
        try:
            W_iou = checkpoint['W_iou']
            print(f', W_IOU:{W_iou:.2f}', end='')
            W_depth = checkpoint['W_depth']
            print(f', W_depth:{W_depth:.2f}')
        except:
            print()

if __name__ == '__main__':
    start = time.time()
    main()
    print('Done, take {} min'.format((time.time()-start)//60))# around 5min