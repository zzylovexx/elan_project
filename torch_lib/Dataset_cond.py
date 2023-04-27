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

# TODO: clean up where this is
def generate_bins(bins): #this case is 2
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

class Dataset(data.Dataset):
    def __init__(self, path, label_path="/label_2/", bins=2, overlap=0.1):

        self.top_label_path = path + label_path
        self.top_img_path = path + "/image_2/"
        self.top_calib_path = path + "/calib/"
        # use a relative path instead?

        # TODO: which camera cal to use, per frame or global one?
        self.proj_matrix = get_P(os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/camera_cal/calib_cam_to_cam.txt'))

        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1,bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0,bins):
            self.bin_ranges.append(( (i*self.interval - overlap) % (2*np.pi), \
                                (i*self.interval + self.interval + overlap) % (2*np.pi)) )

        # hold average dimensions
        class_list = ['Car', 'Van', 'Truck', 'Pedestrian','Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)

        self.object_list = self.get_objects(self.ids)

        # pre-fetch all labels
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
        # P doesn't matter here
        obj = DetectedObject(self.curr_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label)
        
        #theta_ray_condition
        cond = torch.tensor(obj.theta_ray).expand(1, obj.img.shape[1], obj.img.shape[2])
        img_cond = torch.concat((obj.img, cond), dim=0) # 3+1, 224, 224 + grouploss看看
        return img_cond, label

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

        Alpha = line[3] # what we will be regressing
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
        # modify for the average
        Dimension -= self.averages.get_item(Class)

        Location = [line[11], line[12], line[13]] # x, y, z
        Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)
        
        # alpha is [-pi..pi], shift it to be [0..2pi]
        angle = Alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1
        
        calib_path = self.top_calib_path + '%s.txt'%id
        cam_to_img = get_calibration_cam_to_image(calib_path)
        Offset = np.array(calc_center_offset_ratio(Box_2D, Location, cam_to_img))
        
        if len(line) == 16:
            Group = line[15] #line[-1]
            label = {
                    'Class': Class,
                    'Box_2D': Box_2D,
                    'Dimensions': Dimension,
                    'Alpha': Alpha,
                    'Orientation': Orientation,
                    'Confidence': Confidence,
                    'Location': Location,
                    'Center_Offset': Offset,
                    'Group': Group
                    }
        else:
            label = {
                    'Class': Class,
                    'Box_2D': Box_2D,
                    'Dimensions': Dimension,
                    'Alpha': Alpha,
                    'Orientation': Orientation,
                    'Confidence': Confidence,
                    'Location': Location,
                    'Center_Offset': Offset,
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

                Alpha = line[3] # what we will be regressing
                Ry = line[14]
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]

                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

                buf.append({
                        'Class': Class,
                        'Box_2D': Box_2D,
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
                box_2d = label['Box_2D']
                detection_class = label['Class']
                objects.append(DetectedObject(img, detection_class, box_2d, proj_matrix, label=label))

            data[id]['Objects'] = objects

        return data


"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""
class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        if isinstance(proj_matrix, str): # filename
            proj_matrix = get_P(proj_matrix)
            # proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class
        #self.center_offset = self.calc_center_offset(label['Box_2D'], label['Location'], proj_matrix)
        #self.center_offset_ratio = self.calc_center_offset_ratio(label['Box_2D'], label['Location'], proj_matrix)

    def calc_theta_ray(self, img, box_2d, proj_matrix):#透過跟2d bounding box 中心算出射線角度
        width = img.shape[1]
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

    def format_img(self, img, box_2d):

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
        pt1 = box_2d[0]
        pt2 = box_2d[1]
        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # recolor, reformat
        batch = process(crop)

        return batch
    
    #offset pixels
    def calc_center_offset(self, d2_box, d3_location, cam_to_img, resize=224):
        d2_center = get_2d_center(d2_box)
        proj_center = project_3d_pt(d3_location, cam_to_img)
        d2_box_size = get_box_size(d2_box)
        #resize factor
        factor_x = resize / d2_box_size[0] # transform.resize to 224
        factor_y = resize / d2_box_size[1] 
        
        offset_x = (proj_center[0] - d2_center[0]) * factor_x
        offset_y = (proj_center[1] - d2_center[1]) * factor_y
        
        # delta out of range
        if abs(offset_x) > resize//2: 
            offset_x = sign(offset_x)*resize//2
        if abs(offset_y) > resize//2:
            offset_y = sign(offset_y)*resize//2
            
        return [math.floor(offset_x), math.floor(offset_y)]
    
    #offset ratio -1~1
    def calc_center_offset_ratio(self, d2_box, d3_location, cam_to_img):
        d2_center = get_2d_center(d2_box)
        proj_center = project_3d_pt(d3_location, cam_to_img)
        d2_box_size = get_box_size(d2_box)
        
        offset_x = (proj_center[0] - d2_center[0]) / float(d2_box_size[0])
        offset_y = (proj_center[1] - d2_center[1]) / float(d2_box_size[1])
            
        return [offset_x, offset_y]
        