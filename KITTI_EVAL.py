import os, torch
from torch_lib.KITTI_Dataset import *
from library.ron_utils import box_depth_error_calculation, calc_depth_with_alpha_theta
import argparse

parser = argparse.ArgumentParser()

# path setting
parser.add_argument("--weights-path", '-W_PATH', required=True, help='weighs path')
parser.add_argument("--result-path", '-R_PATH', required=True, help='path (folder name) of the generated pred-labels')

# kitti evaluation should be with 3dAP, bevAP functions
def ron_evaluation(val_ids, diff_list, cls_list, result_root, gt_root='Kitti/training/label_2'):
    GT_dim = list()
    GT_depth = list()
    GT_alpha = list()
    REG_dim = list()
    REG_depth = list()
    REG_alpha = list()
    for id_ in val_ids:
        gt_label = os.path.join(gt_root, f'{id_}.txt')
        gt_objects = [Object3d(line) for line in open(gt_label).readlines()]
        reg_label = os.path.join(result_root, f'{id_}.txt')
        reg_objects = [Object3d(line) for line in open(reg_label).readlines()]
        count = 0
        for i in range(len(gt_objects)):
            gt = gt_objects[i]
            if gt.cls_type.lower() in cls_list and gt.level in diff_list:
                #print(gt.alpha, reg.alpha)
                reg = reg_objects[count]
                count += 1
                GT_dim.append(gt.dim)
                GT_depth.append(gt.pos[2])
                GT_alpha.append(gt.alpha)

                REG_dim.append(reg.dim)
                REG_depth.append(reg.pos[2])
                REG_alpha.append(reg.alpha)
                #reg_calc_depth = calc_depth_with_alpha_theta(img2.shape[1], obj.box2d, cam_to_img.p2, reg_dim[1], reg_dim[2], reg_alpha)
        
    GT_dim = np.array(GT_dim)
    GT_depth = np.array(GT_depth)
    GT_alpha = np.array(GT_alpha)
    REG_dim = np.array(REG_dim)
    REG_depth = np.array(REG_depth)
    REG_alpha = np.array(REG_alpha)

    alpha_diff = np.cos(GT_alpha - REG_alpha)
    dim_diff = np.mean(abs(GT_dim - REG_dim), axis=0)
    print(f'[Alpha diff] abs_mean: {1-alpha_diff.mean():.4f}')
    print(f'[DIM diff] H:{dim_diff[0]:.4f}, W:{dim_diff[1]:.4f}, L:{dim_diff[2]:.4f}')
    print('[Depth error]')
    box_depth_error_calculation(GT_depth, REG_depth, 5)

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
    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_path
    result_root = FLAGS.result_path
    split = FLAGS.split
    device = torch.device('cuda:0')
    checkpoint = torch.load(weights_path, map_location=device) #if training on 2 GPU, mapping on the same device
    cfg = checkpoint['cfg']
    diff_list = cfg['diff_list']
    cls_list = cfg['class_list']
    val_ids = [x.strip() for x in open('Kitti/ImageSets/val.txt').readlines()]
    print('=============[VAL EVAL]===============')
    ron_evaluation(val_ids, diff_list, cls_list, result_root)
    train_ids = [x.strip() for x in open('Kitti/ImageSets/train.txt').readlines()]
    print('=============[Train EVAL]===============')
    ron_evaluation(train_ids, diff_list, cls_list, result_root)
    '''
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
    '''