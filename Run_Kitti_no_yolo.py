"""
This script will use the 2D box from the label rather than from YOLO,
but will still use the neural nets to get the 3D position and plot onto the
image. Press space for next image and escape to quit
"""
from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages

import os
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weight-dir', default='weights_group', help='path to the Kiiti weights folder')
parser.add_argument('--result-path', default='GT_group', help='path to put in the generated txt (suggest name after date i.e. GT_orient_0228)')

def plot_regressed_3d_bbox(img, truth_img, cam_to_img, box_2d, dimensions, alpha, theta_ray):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box_2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location, orient

def main():
    FLAGS = parser.parse_args()
    
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/' + FLAGS.weight_dir # '/weights_orient'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    result_path= FLAGS.result_path
    os.makedirs(result_path,exist_ok=True)

    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print ('Using previous model %s'%(weights_path+model_lst[-1]))
        my_vgg = vgg.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # defaults to /eval
    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/eval')
    #dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training')
    averages = ClassAverages.ClassAverages()

    all_images = dataset.all_objects()
    for key in sorted(all_images.keys()):

        start_time = time.time()

        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']
        
        #added
        lines = list()
        for detectedObject in objects:
            label = detectedObject.label

            if label['Class'] == 'DontCare':
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img
            input_tensor.cuda()

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(label['Class'])

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += dataset.angle_bins[argmax]
            alpha -= np.pi

            location, rotation_y = plot_regressed_3d_bbox(img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)

            #print('Estimated pose: %s'%location)
            #print('Truth pose: %s'%label['Location'])
            #print('-------------')
            lines+=f"{label['Class']} 0.0 0 {alpha:.2f} {label['Box_2D'][0][0]} {label['Box_2D'][0][1]} {label['Box_2D'][1][0]} {label['Box_2D'][1][1]} {dim[0]:.2f} {dim[1]:.2f} {dim[2]:.2f} {location[0]:.2f} {location[1]:.2f} {location[2]:.2f} {rotation_y:.2f} {1.00} \n"


        #print('Got %s poses in %.3f seconds\n'%(len(objects), time.time() - start_time))

        #write to txt
        print(key)
        with open(f'{result_path}/{key}.txt','w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    main()
