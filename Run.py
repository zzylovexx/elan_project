"""
目前搭配dataset_theta為monodle之分類angle形式
"""


from torch_lib.Dataset_monodle import *
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


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray,detectionid=None, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)#透過數學找到xyz（location)

    orient = alpha + theta_ray
    # print('orient:',orient)
    if img_2d is not None:
        plot_2d_box(img_2d, box_2d,detectionid)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location,orient
def class2angle(bin_class,residual):
    # angle_per_class=2*torch.pi/float(12)
    angle_per_class=2*np.pi/float(num_heading_bin)
    angle=float(angle_per_class*bin_class)
    angle=angle+residual
    # print(angle)
    return angle

def main():

    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]

    weight_abs_path='/home/chang0731/Desktop/elan_project/3D-BoundingBox/weights_cond_test/epoch_40.pkl'
    
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        #print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        #checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        checkpoint=torch.load(weight_abs_path)
        print('use previous weight:',weight_abs_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

   

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video:
        if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "eval/video/2011_09_26/image_2/"
            cal_dir = "camera_cal/"

    
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
        yolo_img = np.copy(truth_img)
        detections,confidences = yolo.detect(yolo_img)
        #print('value',confidences)
        lines=[]
        for detectionid,detection in enumerate(detections):

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
            cond = torch.tensor(theta_ray).expand(1, input_img.shape[1], input_img.shape[2])
            img_cond = torch.concat((input_img, cond), dim=0) # 3+1, 224, 224 + grouploss看看
            input_tensor = torch.zeros([1,4,224,224]).cuda()
            input_tensor[0,:,:,:] = img_cond #除了batch其他dim都配原圖資訊進去
           
            [orient, conf, dim] = model(input_tensor)
            
            
            orient = orient.cpu().data.numpy()[0, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)
            

            
            #print('dim:',dim)
            cls_argmax = np.argmax(conf)
            resdiual_orient = orient[cls_argmax] 
            # print(cls_argmax)
            # print('theta_ray:',theta_ray)
            # print('resdiual_orient:',resdiual_orient)
            alpha=class2angle(cls_argmax,resdiual_orient)
            # print('alpha:',alpha)
            if alpha >np.pi:
                alpha-=(2*np.pi)
           
            if FLAGS.show_yolo:
                location,_ = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray,detectionid, truth_img)
                #this location means object center ,but in kitti lable it label th buttom center of objet ,so the y location need add 1/2 height
            else:
                location,rotation_y = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray,detectionid)
            location_kitti=location
            location_kitti[1]=location_kitti[1]+dim[0]*0.5
            
            lines+=f"{detection.detected_class} 0.0 0 {alpha:.2f} {box_2d[0][0]} {box_2d[0][1]} {box_2d[1][0]} {box_2d[1][1]} {dim[0]:.2f} {dim[1]:.2f} {dim[2]:.2f} {location_kitti[0]:.2f} {location_kitti[1]:.2f} {location_kitti[2]:.2f} {rotation_y:.2f} {confidences[detectionid]:.2f}\n"
             
            if not FLAGS.hide_debug:
                print('Estimated pose : %s'%location)
                #print('Estimated pose kitti: %s'%location_kitti)
        pic_dir='look/'
        os.makedirs(pic_dir,exist_ok=True)
        
        if (img_id == '0000000100'):
            break

        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            #cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
            #print(pic_dir+img_file)
            #cv2.imwrite(pic_dir+img_id +".png",numpy_vertical)
        else:
            pass
            # cv2.imshow('3D detections', img)
        
        
        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')
            print(img_id)
        result_path='./temp/'
        os.makedirs(result_path,exist_ok=True)
       
       # write to txt
        with open(f'{result_path}{img_id}.txt','w') as f:
            f.writelines(lines)
            
        
        if FLAGS.video:
            cv2.waitKey(1)
        # else:
        #     if cv2.waitKey(0) != 32: # space bar
        #         exit()

if __name__ == '__main__':
    main()
