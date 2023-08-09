"""
要修改Dataset_4dim 轉換為monodle形式
"""



from PyTorch_YOLOv3_kitti.detect2 import yolo_kitti
#from PyTorch_YOLOv3_kitti.models import *

from torch_lib.Dataset_4dim import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, Model_center, ClassAverages
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

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", default=True,
                    help="Supress the printing of each 3d location")

parser.add_argument("--weights-path", required=True,
                    help="abs path for weights")

parser.add_argument("--save-path", default=True,
                    help="Save path for the generated label")


def plot_regressed_3d_bbox(img, cam_to_img, box2d, dimensions, alpha, theta_ray,detectionid, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box2d, alpha, theta_ray)#透過數學找到xyz（location)

    orient = alpha + theta_ray
    # print('orient:',orient)
    if img_2d is not None:
        plot_2d_box(img_2d, box2d,detectionid)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location,orient

def main():

    FLAGS = parser.parse_args()

    # load torch
    weight_abs_path= FLAGS.weights_path
    #weight_abs_path = 'weights_group/epoch_20_b16_cos_1.pkl'

    my_vgg = vgg.vgg19_bn(pretrained=True)
    #0407 for cond
    #my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
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

    
    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    #calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    #calib_file = calib_path + "calib_cam_to_cam.txt"
    
    # using P from each frame
    calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png" #elan:jpg,kitti:png
       

        # P for each frame
        calib_file = get_calibration_cam_to_image(calib_path + img_id + ".txt")
    
        truth_img = cv2.imread(img_file)
        
        img = np.copy(truth_img)
        
        
        detections,confidences = kitti_yolo_model.detect(img_file)#input is file path not cv2.imread
        
        lines=[]
        for detectionid, detection in enumerate(detections):
            #print(detection.detected_class)
            #print(detection.box2d)
            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box2d, calib_file)    
            except:
                continue
            

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box2d = detection.box2d
           
            detected_class = detection.detected_class
            #print('detectionclass:', detection.detected_class)
            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,0:3,:,:] = input_img #除了batch其他dim都配原圖資訊進去
            
            #for cond
            #input_tensor = torch.zeros([1,4,224,224]).cuda()
            #input_tensor[0,0:3,:,:] = input_img #除了batch其他dim都配原圖資訊進去
            #input_tensor[0,3,:,:] = torch.tensor(theta_ray).expand(1,224,224) #embed

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
                location,_ = plot_regressed_3d_bbox(img, proj_matrix, box2d, dim, alpha, theta_ray, detectionid, truth_img)
                #this location means object center ,but in kitti lable it label th buttom center of objet ,so the y location need add 1/2 height
            else:
                location,rotation_y = plot_regressed_3d_bbox(img, proj_matrix, box2d, dim, alpha, theta_ray, detectionid)
            location_kitti=location
            location_kitti[1]=location_kitti[1]+dim[0]*0.5
            
            lines+=f"{detection.detected_class} 0.0 0 {alpha:.2f} {box2d[0][0]} {box2d[0][1]} {box2d[1][0]} {box2d[1][1]} {dim[0]:.2f} {dim[1]:.2f} {dim[2]:.2f} {location_kitti[0]:.2f} {location_kitti[1]:.2f} {location_kitti[2]:.2f} {rotation_y:.2f} {confidences[detectionid]:.2f} \n"

            if not FLAGS.hide_debug:
                #print('Estimated pose : %s'%location)
                print('Estimated pose kitti: %s'%location_kitti)
        '''
        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            cv2.imshow('3D detections', img)
        '''
        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')
            print(img_id)

        result_path= FLAGS.save_path
        os.makedirs(result_path,exist_ok=True)
       
        #write to txt
        #print(lines)
        with open(os.path.join(result_path, f'{img_id}.txt'),'w') as f:
            f.writelines(lines)

    print('Done')

if __name__ == '__main__':
    main()
