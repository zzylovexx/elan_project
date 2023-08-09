from .models import *
from .utils.utils import *
from .utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator




class_path='PyTorch_YOLOv3_kitti/data/kitti.names'

batch_size=1
n_cpu=8
img_size=416

cuda = torch.cuda.is_available() 
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#model = Darknet(config_path, img_size=img_size)
#model.load_weights(weights_path)

# print('model path: ' +weights_path)
# if cuda:
#     model.cuda()
#     print("using cuda model")

# model.eval() # Set in evaluation mode

classes = load_classes(class_path)
class yolo_kitti():
    def __init__(self,config_path ='PyTorch_YOLOv3_kitti/config/yolov3-kitti.cfg',kitti_weights = 'PyTorch_YOLOv3_kitti/weights/yolov3-kitti.weights'):
        
        self.model = Darknet(config_path, img_size=416)
        self.model.load_weights(kitti_weights)
        self.model.cuda()
        self.model.eval()

    def detect(self,img_path):
        box2d=[]
        confidences=[]
        detected_class=[]
        detections=[]
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
            # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
            # Add padding
        
        img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
            # Resize and normalizel
    
        input_img = resize(img, (416,416,3), mode='reflect')
            # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
            # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        # input_img=torch.from_numpy(input_img)
        input_imgs = Variable(input_img.type(Tensor)).unsqueeze(0)
        #input_imgs=input_imgs.permute(0,3,1,2)
        with torch.no_grad():
            output=self.model(input_imgs)
            output = non_max_suppression(output, 80, conf_thres=0.8, nms_thres=0.5)
        for detection in output:
        # pred_boxes = detection[:, :5].cpu().numpy()
        # scores = detection[:, 4].cpu().numpy()
        # pred_labels = detection[:, -1].cpu().numpy()
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                pred_boxes=x1, y1, x2, y2
                scores=conf
                pred_labels=cls_pred
                # print(pred_boxes)
                # print(classes[int(pred_labels)])
                top_left=(int((x1*1242/416).cpu().numpy().item()),int((y1*1242/416-433).cpu().numpy().item()))
                buttom_right=(int((x2*1242/416).cpu().numpy().item()),int((y2*1242/416-433).cpu().numpy().item()))
                #self.box2d.append((top_left,buttom_right))
                #self.detected_class.append(classes[int(pred_labels)])
                confidences.append(cls_conf.cpu().numpy().item())
                class_=classes[int(pred_labels)]
                detections.append(Detection([top_left,buttom_right],class_))
        return detections,confidences

class Detection():
    def __init__(self,box2d,class_):
        self.box2d=box2d
        self.detected_class=class_