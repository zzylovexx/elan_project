import os
import cv2
import glob
import math
import numpy as np
import argparse

from library.File import *
from jenkspy import jenks_breaks

parser = argparse.ArgumentParser()

parser.add_argument("--dataset-dir", default="Kitti/training/", help="path to Kitti/training")
parser.add_argument("--group-label-dir", default="label_2_group/", help="group label folder name")
parser.add_argument("--calib-dir", default="camera_cal/", help="path to the camera_cal")
parser.add_argument("--calib-file", default="calib_cam_to_cam.txt", help="file name of the camera_cal.txt")

def main():

    FLAGS = parser.parse_args()
    # Kitti image_2 dir / label_2 dir
    img_root = os.path.join(FLAGS.dataset_dir, 'image_2')
    imgs = glob.glob(os.path.join(img_root, '*.png'), recursive=True)
    label_root = os.path.join(FLAGS.dataset_dir, 'label_2')
    files = glob.glob(os.path.join(label_root, '*.txt'), recursive=True)
    # Kitti camera cal dir
    calib_file = os.path.join(FLAGS.calib_dir, FLAGS.calib_file)
    proj_matrix = get_P(calib_file)
    dir_name = FLAGS.group_label_dir
    os.makedirs(os.path.join(FLAGS.dataset_dir, dir_name), exist_ok=True)

    obj_count = 0
    for i in range(len(files)):
        img = cv2.imread(imgs[i])
        width = img.shape[1]
        with open(files[i]) as f:
            orients = list()
            new_data = list()
            lines = f.readlines()
            for idx, line in enumerate(lines):
                elements = line.split()
                if elements[0] == 'DontCare':
                    continue
                else:
                    for j in range(1, len(elements)):
                        elements[j] = float(elements[j])

                    
                    top_left = (int(round(elements[4])), int(round(elements[5])))
                    btm_right = (int(round(elements[6])), int(round(elements[7])))
                    theta_ray = calc_theta_ray(width, (top_left, btm_right), proj_matrix)

                    alpha = elements[3]
                    ty = alpha + theta_ray 
                    orients.append(math.cos(ty))
                    #rotation_y = elements[14]
                    #orients.append(rotation_y)
            
            if len(orients) > 1:
                clustered = clustering(orients, 0.7)
                # shift group_idx:
                for k in range(len(clustered)):
                    clustered[k] += obj_count
                obj_count = max(clustered)
            else:
                obj_count +=1
                clustered = [obj_count]
                
            # DontCare append -1
            for j in range(len(lines)-len(clustered)):
                clustered.append(-1)
                
            # add grouping info
            for c, line in zip(clustered, lines):
                new_data.append(line[:-1] + ' ' + str(c))
        
            # write new .txt
            with open(files[i].replace('label_2', dir_name), 'w') as new_f:
                for data in new_data:
                    new_f.writelines(data+'\n')
    
    print('Group labels.txt are saved in %s'%(os.path.join(FLAGS.dataset_dir, dir_name)))

def clustering(orient, threshold=0.7):
    array = np.array(orient)
    gvf = 0.0
    nclasses = 1
    try:
        while gvf < threshold:
            nclasses += 1
            gvf = goodness_of_variance_fit(array, nclasses)
            #print(nclasses, gvf)
        breaks = jenks_breaks(array, nclasses)
    except:
        breaks = jenks_breaks(array, nclasses-1)

    breaks = list(dict.fromkeys(breaks))
    #group by the closest breakpoint
    group = list()
    for ele in orient:
        minimum = 999
        for i, b in enumerate(breaks):
            if ele == b:
                idx = i+1
                break
            else:
                tmp = (ele-b)**2
                if tmp < minimum:
                    minimum = tmp
                    idx = i+1
        group.append(idx)
    return group

def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jenks_breaks(array, classes)
    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # max value of zones
    maxz = max(classified)
    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)
    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort if len(classified)!=0])
    # goodness of variance fit (evaluate the performance of clustering)
    gvf = (sdam - sdcm) / sdam
    return gvf
 
def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks)-1

def calc_theta_ray(width, box_2d, proj_matrix):
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = (box_2d[1][0] + box_2d[0][0]) / 2
    dx = center - (width / 2)
    
    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
    angle = angle * mult
    return angle

if __name__=='__main__':
    main()