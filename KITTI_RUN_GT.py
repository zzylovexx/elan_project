from torchvision.models import vgg
from torch_lib.Model_heading_bin import *
from torchvision import transforms
from torch_lib.KITTI_Dataset import *
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

def main():
    #weights_path = 'weights/0808car/KITTI_BL_B4_50.pkl'
    #result_root = '0808car/KITTI_BL_B4_50'
    #os.makedirs(result_root, exist_ok=True)
    #device = torch.device('cuda:0') # 選gpu的index

    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_path
    result_root = FLAGS.result_path
    os.makedirs(result_root, exist_ok=True)

    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    checkpoint = torch.load(weights_path, map_location=device) #if training on 2 GPU, mapping on the same device
    cfg = checkpoint['cfg']
    bin_num = cfg['bins'] 
    is_cond = cfg['cond']
    diff_list = cfg['diff_list']
    cls_list = cfg['class_list']
    
    angle_per_class = 2*np.pi/float(bin_num)

    my_vgg = vgg.vgg19_bn(weights='DEFAULT')
    if is_cond:
        print("< add Condition (4-dim) as input >")
        my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
    model = Model(features=my_vgg.features, bins=bin_num).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # for img processing
    process = transforms.Compose([transforms.ToTensor(), 
                                  transforms.Resize([224,224], transforms.InterpolationMode.BICUBIC), 
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    dataset_train = KITTI_Dataset(cfg, process, split='train')    
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
                reg_labels += obj.REG_result_to_kitti_format_label(reg_alpha, reg_dim, reg_pos) + '\n'
                # [TRAIN]
                if id_ in train_ids:
                    train_GT_depth.append(obj.pos[2])
                    train_CALC_depth.append(calc_depth_with_alpha_theta(img2.shape[1], obj.box2d, cam_to_img.p2, reg_dim[1], reg_dim[2], reg_alpha))
                # [VAL] compare GT_depth and CALC_depth
                if id_ in val_ids:
                    val_GT_depth.append(obj.pos[2])
                    val_CALC_depth.append(calc_depth_with_alpha_theta(img2.shape[1], obj.box2d, cam_to_img.p2, reg_dim[1], reg_dim[2], reg_alpha))
        
        with open(os.path.join(result_root, f'{id_}.txt'), 'w') as f:
            f.writelines(reg_labels)
    
    # eval part
    train_GT_depth = np.array(train_GT_depth)
    train_CALC_depth = np.array(train_CALC_depth)
    val_GT_depth = np.array(val_GT_depth)
    val_CALC_depth = np.array(val_CALC_depth)
    ron_evaluation(val_ids, diff_list, cls_list, result_root)
    print('[MY CALC Depth error]')
    box_depth_error_calculation(val_GT_depth, val_CALC_depth, 5)
    #write as file as well
    org_stdout = sys.stdout
    os.makedirs(f'KITTI_eval/{result_root.split("/")[0]}', exist_ok=True)
    f = open(f'KITTI_eval/{result_root}.txt', 'w')
    sys.stdout = f
    print_info(checkpoint, cfg)
    ron_evaluation(val_ids, diff_list, cls_list, result_root)
    print('[MY CALC Depth error]')
    box_depth_error_calculation(val_GT_depth, val_CALC_depth, 5)
    print('=============[Train EVAL]===============')
    ron_evaluation(train_ids, diff_list, cls_list, result_root)
    print('[MY CALC Depth error]')
    box_depth_error_calculation(train_GT_depth, train_CALC_depth, 5)
    sys.stdout = org_stdout
    f.close()
    print(f'save in KITTI_eval/{result_root}.txt')
    
def print_info(ckpt, cfg):
    class_list = cfg['class_list']
    diff_list = cfg['diff_list']
    group = cfg['group']
    cond = cfg['cond']
    W_consist = ckpt['W_consist']
    W_ry = ckpt['W_ry']
    W_group = ckpt['W_group']
    print('Class:', class_list, end=', ')
    print('Diff:', diff_list, end=', ')
    print(f'Group:{group}, cond:{cond}', end=' ')
    print(f'[Weights] W_consist:{W_consist:.2f}, W_ry:{W_ry:.2f}, W_group:{W_group:.2f}', end='')
    try:
        W_depth = ckpt['W_depth']
        print(f', W_depth:{W_depth:.2f}')
    except:
        print()

if __name__ == '__main__':
    start = time.time()
    main()
    print('Done, take {} min'.format((time.time()-start)//60))# around 5min