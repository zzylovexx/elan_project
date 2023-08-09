import cv2
import numpy as np
import os
import random
import math

import torch
from torchvision import transforms
from torch.utils import data

from library.File import *
from library.Plotting import *
from library.ron_utils import *

from .ClassAverages import ClassAverages


'''
orient 分類與loss使用monodle 之類似型式
使用上需要確認model ouput dim
目前配合train_cond.py以及run.py

'''
def angle2class(angle, num_heading_bin):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_heading_bin) #degree:30 radius:0.523
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2) #residual 有正有負

    return class_id, residual_angle

class Dataset(data.Dataset):
    def __init__(self, path, split='train', camera='left', num_heading_bin=4, condition=False):
        
        
        if camera.lower()=='left':
            self.top_label_path = path + '/label_2/'
            self.top_img_path = path + "/image_2/"
        elif camera.lower()=='right':
            self.top_label_path = path + '/label_3/'
            self.top_img_path = path + "/image_3/"
        self.top_calib_path = path + "/calib/"
        #self.extra_label_path = path + '/extra_label/' #using generated extra label
        self.num_heading_bin = num_heading_bin
        self.condition = condition
        split_dir = os.path.join(path, 'ImageSets', split + '.txt')
        self.ids = [x.strip() for x in open(split_dir).readlines()]

        # TODO: which camera cal to use, per frame or global one?
        self.proj_matrix = get_P(os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/camera_cal/calib_cam_to_cam.txt'))

        # hold average dimensions
        class_list = ['Car', 'Van', 'Truck', 'Pedestrian','Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)

        self.object_list = self.get_objects(self.ids)

        
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None

    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + '%s.png'%id)

        label = self.labels[id][str(line_num)]
        obj = DetectedObject(self.curr_img, label['Class'], label['Box2d'], self.proj_matrix, label=label)
        #label['Depth_bias'] = obj.get_depth_bias()
        label['Theta_ray'] = obj.theta_ray
        if self.condition:
            #cond = torch.tensor(obj.theta_ray).expand(1, obj.img.shape[1], obj.img.shape[2])
            #cond = torch.tensor(obj.boxH_ratio).expand(1, obj.img.shape[1], obj.img.shape[2]) #box height ratio         
            cond = torch.tensor(obj.Y_axis_ratio).expand(1, obj.img.shape[1], obj.img.shape[2]) #Y coord ratio, 0706 added
            img_cond = torch.concat((obj.img, cond), dim=0)
            return img_cond, label
        
        return obj.img, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                for line_num,line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue

                    dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)

                    objects.append((id, line_num))


        self.averages.dump_to_file()
        return objects


    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt'%id).read().splitlines()
        label = self.format_label(lines[line_num], id)
        '''
        extra_labels = get_extra_labels(self.extra_label_path + '%s.txt'%id)
        label['Group'] = extra_labels[line_num]['Group_Ry']
        label['Theta'] = extra_labels[line_num]['Theta_ray']
        label['boxH_ratio'] = extra_labels[line_num]['Box_H'] / 224.0
        '''
        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line, id):
        line = line.split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        Truncate = line[1] # truncate ratio
        Alpha = line[3] # what we will be regressing
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        box2d = [top_left, bottom_right]

        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
        # modify for the average
        Dimension -= self.averages.get_item(Class)

        Location = [line[11], line[12], line[13]] # x, y, z
        Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

        heading_class,heading_residual=angle2class(Alpha, self.num_heading_bin)
        #theta用算的 因為label_3的ry和alpha和label_2的相同

        label = {
                'Class': Class,
                'Truncate': Truncate,
                'Box2d': box2d,
                'Dimensions': Dimension,
                'Alpha': Alpha,
                'Location': Location,
                'heading_residual': heading_residual,
                'heading_class': heading_class,
                'Ry': Ry
                }
        return label

    # will be deprc soon
    def parse_label(self, label_path):
        buf = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Truncate = line[1] # truncate ratio
                Alpha = line[3] # what we will be regressing
                Ry = line[14]
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                box2d = [top_left, bottom_right]

                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

                buf.append({
                        'Class': Class,
                        'Truncate': Truncate,
                        'Box2d': box2d,
                        'Dimensions': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry
                    })
        return buf

    # will be deprc soon
    def all_objects(self):
        data = {}
        for id in self.ids:
            data[id] = {}
            img_path = self.top_img_path + '%s.png'%id
            img = cv2.imread(img_path)
            data[id]['Image'] = img

            # using p per frame
            calib_path = self.top_calib_path + '%s.txt'%id
            proj_matrix = get_calibration_cam_to_image(calib_path)

            # using P_rect from global calib file
            #proj_matrix = self.proj_matrix (command out 0406 for drawing correct 3d box

            data[id]['Calib'] = proj_matrix

            label_path = self.top_label_path + '%s.txt'%id
            labels = self.parse_label(label_path)
            objects = []
            for label in labels:
                box2d = label['Box2d']
                detection_class = label['Class']
                objects.append(DetectedObject(img, detection_class, box2d, proj_matrix, label=label))

            data[id]['Objects'] = objects

        return data

"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""
class DetectedObject:
    def __init__(self, img, detection_class, box2d, proj_matrix, label=None):

        if isinstance(proj_matrix, str): # filename
            proj_matrix = get_P(proj_matrix)
            # proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.img_W = img.shape[1]
        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box2d, proj_matrix)
        self.img = self.format_img(img, box2d)
        self.box2d = box2d
        self.label = label
        self.detection_class = detection_class
        self.boxH_ratio = get_box_size(box2d)[1] / 224.
        self.Y_axis_ratio = get_box_center(box2d)[1] / img.shape[0]
        
        self.averages = ClassAverages([],'class_averages.txt')
        #print(self.averages.dimension_map)
        

    def calc_theta_ray(self, img, box2d, proj_matrix):#透過跟2d bounding box 中心算出射線角度
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box2d[1][0] + box2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box2d):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        process = transforms.Compose ([
            transforms.ToTensor(),
            #transforms.Grayscale(num_output_channels=3), #make it to gray scale
            normalize
        ])
        
        # crop image
        pt1 = box2d[0]
        pt2 = box2d[1]
        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # recolor, reformat
        batch = process(crop)

        return batch
    
    #offset ratio -1~1
    def calc_offset_ratio(self, box2d, d3_location, cam_to_img):
        return calc_center_offset_ratio(box2d, d3_location, cam_to_img)
    
    def get_depth_bias(self):
        if self.label != None:
            return self.calc_depth_bias(self.img_W, self.box2d, self.proj_matrix, self.label)
        else:
            RuntimeError('No label')

    #img.shape[1], box2d, proj_matrix, label
    def calc_depth_bias(self, img_W, box2d, cam_to_img, label):
        obj_W = self.averages.get_item(self.detection_class)[1] + label['Dimensions'][1]
        obj_L = self.averages.get_item(self.detection_class)[2] + label['Dimensions'][2]
        alpha = label['Alpha']
        trun = label['Truncate'] 
        depth_GT = label['Location'][2]
        depth_calc = calc_depth_with_alpha_theta(img_W, box2d, cam_to_img, obj_W, obj_L, alpha, trun)
        #print(img_W, box2d, cam_to_img, obj_W, obj_L, alpha, trun)
        #print('GT',depth_GT)
        #print('calc',depth_calc)
        return depth_GT - depth_calc