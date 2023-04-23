"""
This script will use the 2D box from the label rather than from YOLO,
but will still use the neural nets to get the 3D position and plot onto the
image. Press space for next image and escape to quit
"""
from torch_lib.Dataset_4dim import *
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
parser.add_argument('--weight-path', default='weights/epoch_20.pkl', help='path to the weights')
parser.add_argument('--result-path', default='baseline', help='path to put in the generated txt (suggest name after date i.e. GT_orient_0228)')

def regress_location_orient(cam_to_img, box_2d, dimensions, alpha, theta_ray):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    return location, orient

def main():
    FLAGS = parser.parse_args()
    
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/' + FLAGS.weight_path
    result_path= FLAGS.result_path
    os.makedirs(result_path, exist_ok=True)
    print ('Using model %s'%(weights_path))
    my_vgg = vgg.vgg19_bn(pretrained=True)
    my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
    model = Model.Model(features=my_vgg.features, bins=2).cuda()
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training')
    averages = ClassAverages.ClassAverages()
    print('Loading all objects')
    all_images = dataset.all_objects()
    print('Start loop')
    for key in sorted(all_images.keys()):

        data = all_images[key]
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

            #for cond
            input_tensor = torch.zeros([1,4,224,224]).cuda()
            input_tensor[0,0:3,:,:] = input_img #除了batch其他dim都配原圖資訊進去
            input_tensor[0,3,:,:] = torch.tensor(theta_ray).expand(1,224,224) #embed

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

            location, rotation_y = regress_location_orient(cam_to_img, label['Box_2D'], dim, alpha, theta_ray)

            lines+=f"{label['Class']} 0.0 0 {alpha:.2f} {label['Box_2D'][0][0]} {label['Box_2D'][0][1]} {label['Box_2D'][1][0]} {label['Box_2D'][1][1]} {dim[0]:.2f} {dim[1]:.2f} {dim[2]:.2f} {location[0]:.2f} {location[1]:.2f} {location[2]:.2f} {rotation_y:.2f}\n"


        print(key)

        #write to txt
        with open(f'{result_path}/{key}.txt','w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    main()
