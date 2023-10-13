"""
This script will use the 2D box from the label rather than from YOLO,
but will still use the neural nets to get the 3D position and plot onto the
image. Press space for next image and escape to quit
"""
from torch_lib.Dataset_heading_bin import *
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
parser.add_argument('--weight-dir', default='weights', help='path to the Kiiti weights folder')
parser.add_argument('--weights', required=False, help='The name of weights.pkl')
parser.add_argument('--data-dir', default='Kitti/training', help='path to data folder')

def plot_regressed_3d_bbox(img, truth_img, cam_to_img, box2d, dimensions, alpha, theta_ray):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location, orient

def main():
    
    FLAGS = parser.parse_args()
    '''
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/' + FLAGS.weight_dir
    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model.Model(features=my_vgg.features, bins=2).cuda()
    checkpoint = torch.load(weights_path + '/' + FLAGS.weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # defaults to /eval
    #dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/eval')
    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/' + FLAGS.data_dir)
    '''
    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/' + FLAGS.data_dir)
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
        rys = list()
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

            location, rotation_y = plot_regressed_3d_bbox(img, truth_img, cam_to_img, label['Box2d'], dim, alpha, theta_ray)
            rys.append(f'{rotation_y:.2f}')

        '''
        numpy_vertical = np.concatenate((truth_img, img), axis=0)
        print(key)
        print(rys)
        print('-'*20)
        cv2.imshow(FLAGS.weights, numpy_vertical)
        if cv2.waitKey(0) == 27:
            return
        '''
        
if __name__ == '__main__':
    main()
