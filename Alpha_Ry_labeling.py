import os
import cv2
import glob
import math
import numpy as np
import argparse

from library.File import *
from jenkspy import jenks_breaks

parser = argparse.ArgumentParser()
#0418 added idx*100+group for consider different imgs
parser.add_argument("--label-root", default="Kitti/training/label_2", help="group label folder name")
parser.add_argument("--new-label-dir", default="label_2_new_group", help="new label folder name")

def main():
    FLAGS = parser.parse_args()
    
    label_root = FLAGS.label_root #"Kitti/training/label_2"
    files = glob.glob(os.path.join(label_root, '*.txt'), recursive=True)
    dir_name = FLAGS.new_label_dir
    new_label_root = label_root.replace('label_2', dir_name)
    os.makedirs(new_label_root, exist_ok=True)
    alpha_diff_txt = list()
    ry_diff_txt = list()
    for i in range(len(files)):
        with open(files[i]) as f:
            
            lines = f.readlines()
            new_labels = list()
            alphas = list()

            rys = list()

            for line in lines:
                line = line[:-1]
                elements = line.split()
                if elements[0] == 'DontCare':
                    continue
                else:
                    alphas.append(float(elements[3]))
                    rys.append(float(elements[14]))
                    
            alpha_classes, alpha_diff = get_bin_classes(alphas, num_bin=60)
            ry_classes, ry_diff = get_bin_classes(rys, num_bin=60)
            for line, alpha_class, ry_class in zip(lines, alpha_classes, ry_classes):
                line = line[:-1]
                new_labels.append(line + ' ' + str(100*i+alpha_class) + ' ' + str(100*i+ry_class))

        # write new .txt
        with open(files[i].replace('label_2', dir_name), 'w') as new_f:
            for label in new_labels:
                new_f.writelines(label+'\n')
    
        #record diff(org!=new) txts
        if alpha_diff:
            alpha_diff_txt.append(files[i])
        if ry_diff:
            ry_diff_txt.append(files[i])

    with open('Kitti/training/alpha_diff.txt', 'w') as f:
        for txt in alpha_diff_txt:
            f.writelines(txt+'\n')
    with open('Kitti/training/ry_diff.txt', 'w') as f:
        for txt in ry_diff_txt:
            f.writelines(txt+'\n')

    print('New labels.txt are saved.')

#monodle/blob/main/lib/datasets/utils.py
def angle2class(angle, num_heading_bin=60):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi) 
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    #residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id#, residual_angle

def get_bin_classes(array, num_bin=60):
    org_classes = list()
    for value in array:
        org_classes.append(angle2class(value, num_bin))
    
    info_dict = generate_info_dict(array, org_classes, num_bin)
    info_dict = check_neighbor_classes(info_dict, num_bin)
    new_classes = get_new_classes(info_dict, org_classes)
    diff = 0 if (org_classes == new_classes).all() else 1
    return new_classes, diff
    
def generate_info_dict(array, org_classes, num_bin):
    tmp = [[] for i in range(num_bin)]
    tmp_dict = dict()
    for class_, value in zip(org_classes, array):
        tmp[class_].append(value)
    for idx, list_ in enumerate(tmp):
        if len(list_)==0:
            continue
        tmp_dict[idx] = {'list': list_, 'mean': sum(list_)/len(list_), 'class': idx}
    return tmp_dict

def check_neighbor_classes(dict_, num_bin):
    keys = sorted(dict_.keys())
    angle_per_class = 2*np.pi / num_bin
    # 0-num_bin-1
    for i in range(len(keys)-1):
        if keys[i] == keys[i+1]-1 and abs(dict_[keys[i]]['mean'] - dict_[keys[i+1]]['mean'] < angle_per_class):
            len_i = len(dict_[keys[i]]['list'])
            len_i_1 = len(dict_[keys[i+1]]['list'])

            class_ = dict_[keys[i]]['class'] if len_i>= len_i_1 else dict_[keys[i+1]]['class']
            dict_[keys[i]]['class'] = class_
            dict_[keys[i+1]]['class'] = class_
    # 0 and num_bin-1 is neighbor 
    if 0 in keys and num_bin-1 in keys and abs(dict_[keys[i]]['mean'] - dict_[keys[i+1]]['mean']) < angle_per_class:
        len_i = len(dict_[keys[i]]['list'])
        len_i_1 = len(dict_[keys[i+1]]['list'])
        class_ = dict_[keys[i]]['class'] if len_i>= len_i_1 else dict_[keys[i+1]]['class']
        dict_[keys[i]]['class'] = class_
        dict_[keys[i+1]]['class'] = class_
    return dict_

def get_new_classes(dict_, classes):
    classes = np.array(classes)
    for key in dict_.keys():
        if key != dict_[key]['class']:
            classes[classes==key] = dict_[key]['class']
    return classes

if __name__=='__main__':
    main()