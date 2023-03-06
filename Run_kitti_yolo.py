"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""



from PyTorch_YOLOv3_kitti.detect2 import yolo_kitti
#from PyTorch_YOLOv3_kitti.models import *

from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="Kitti/training/image_2/", #elan_dataset:img, kitti:image_2
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)#透過數學找到xyz（location)

    orient = alpha + theta_ray
    # print('orient:',orient)
    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location,orient

def main():

    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights' #/weights_group
    
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]

    weight_abs_path='./weights_group/epoch_20.pkl' #my weigh_path
    #weight_abs_path = 'weights_group/epoch_20_b16_cos_1.pkl'
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        #print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        #checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        checkpoint=torch.load(weight_abs_path)
        print('use previous weight:',weight_abs_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
   

    kitti_yolo_model=yolo_kitti()#train on kitti_dataset  yolov3
    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video:
        if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "eval/video/2011_09_26/image_2/"
            cal_dir = "eval/video/2011_09_26/"

    
    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"
    
    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png" #elan:jpg,kitti:png
       

        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        
        img = np.copy(truth_img)
        
        
        detections,confidences = kitti_yolo_model.detect(img_file)#input is file path not cv2.imread
       
       
        
        lines=[]
        for detectionid,detection in enumerate(detections):
            # print(detection.detected_class)
            # print(detection.box_2d)
            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d
           
            detected_class = detection.detected_class
            #print('detectionclass:', detection.detected_class)
            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img #除了batch其他dim都配原圖資訊進去

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)
            #print('dim:',dim)
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            #print('alpha:',alpha)
            if FLAGS.show_yolo:
                location,_ = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
                #this location means object center ,but in kitti lable it label th buttom center of objet ,so the y location need add 1/2 height
            else:
                location,rotation_y = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
            location_kitti=location
            location_kitti[1]=location_kitti[1]+dim[0]*0.5
            
            lines+=f"{detection.detected_class} 0.0 0 {alpha:.2f} {box_2d[0][0]} {box_2d[0][1]} {box_2d[1][0]} {box_2d[1][1]} {dim[0]:.2f} {dim[1]:.2f} {dim[2]:.2f} {location_kitti[0]:.2f} {location_kitti[1]:.2f} {location_kitti[2]:.2f} {rotation_y:.2f} {confidences[detectionid]:.2f} \n"
            
            if not FLAGS.hide_debug:
                #print('Estimated pose : %s'%location)
                print('Estimated pose kitti: %s'%location_kitti)

        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        # else:
        #     cv2.imshow('3D detections', img)

        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')
            print(img_id)
        result_path='./20epoch_kitti_yolo/'
        os.makedirs(result_path,exist_ok=True)
       
        #write to txt
        #print(img_id)
        with open(f'{result_path}{img_id}.txt','w') as f:
            f.writelines(lines)
            
        
        if FLAGS.video:
            cv2.waitKey(1)
        # else:
        #     if cv2.waitKey(0) != 32: # space bar
        #         exit()

if __name__ == '__main__':
    main()
