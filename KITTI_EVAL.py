import os, torch
from torch_lib.KITTI_Dataset import *
from library.ron_utils import box_depth_error_calculation, calc_depth_with_alpha_theta
import argparse

parser = argparse.ArgumentParser()

# path setting
parser.add_argument("--weights-path", '-W_PATH', required=True, help='weighs path')
parser.add_argument("--result-path", '-R_PATH', required=True, help='path (folder name) of the generated pred-labels')
parser.add_argument("--split", '-S', default='val', help='train | val | trainval')

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
        for obj in gt_objects:
            if obj.cls_type in cls_list and obj.level in diff_list:
                GT_dim.append(obj.dim)
                GT_depth.append(obj.pos[2])
                GT_alpha.append(obj.alpha)
                
        reg_label = os.path.join(result_root, f'{id_}.txt')
        reg_objects = [Object3d(line) for line in open(reg_label).readlines()]
        for obj in reg_objects:
            if obj.cls_type in cls_list and obj.level in diff_list:
                REG_dim.append(obj.dim)
                REG_depth.append(obj.pos[2])
                REG_alpha.append(obj.alpha)
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

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_path
    result_root = FLAGS.result_path
    split = FLAGS.split
    device = torch.device('cuda:0')
    checkpoint = torch.load(weights_path, map_location=device) #if training on 2 GPU, mapping on the same device
    diff_list = [1, 2] #checkpoint['cfg'] ['diff_list']
    cls_list = ['car'] #checkpoint['cond']['cls_list']
    split_dir = f'Kitti/ImageSets/{split}.txt'
    val_ids = [x.strip() for x in open(split_dir).readlines()]
    ron_evaluation(val_ids, diff_list, cls_list, result_root)