import os, cv2, csv
from torch.utils import data
import numpy as np
from library.ron_utils import angle_correction

def angle2class(angle, bins):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(bins)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle

def class2angle(cls, residual, bins):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * np.pi / float(bins)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    return angle

class KITTI_Dataset(data.Dataset):
    def __init__(self, cfg, process, split='train'):
        path = cfg['path']
        self.label2_path = os.path.join(path, 'label_2')
        self.img2_path = os.path.join(path, 'image_2')
        if split=='train':
            self.label3_path = os.path.join(path, 'label_3')
            self.img3_path = os.path.join(path, 'image_3')
        elif split=='val':
            self.label3_path = self.label2_path
            self.img3_path = self.img2_path

        self.calib_path = os.path.join(path, 'calib') 
        self.extra_label_path = os.path.join(path, 'extra_label') #using generated extra label
        self.bins = cfg['bins']
        self.diff_list = cfg['diff_list']
        self.cls_list = cfg['class_list']
        self.cond = cfg['cond']
        self.split = split
        split_dir = os.path.join('Kitti/ImageSets', split + '.txt')
        self.ids = [x.strip() for x in open(split_dir).readlines()]
        self.cls_dims = dict()
        self.objects_L, self.objects_R, = self.get_objects(self.ids)
        self.targets_L, self.targets_R, = self.get_targets(self.objects_L, self.objects_R)
        self.transform = process
    
    def __len__(self):
        return len(self.objects_L)

    def __getitem__(self, idx):
        # left_obj, left_label, right_obj, right_label
        if self.split=='train':
            return self.transform(self.objects_L[idx].crop), self.targets_L[idx],  \
                   self.transform(self.objects_R[idx].crop), self.targets_R[idx]
        else:
            return self.transform(self.objects_L[idx].crop), self.targets_L[idx]

    def get_objects(self, ids):
        all_objects_L = list()
        all_objects_R = list()
        for id_ in ids:
            label2_txt = os.path.join(self.label2_path, f'{id_}.txt')
            label3_txt = os.path.join(self.label3_path, f'{id_}.txt')
            cam_to_img = FrameCalibrationData(os.path.join(self.calib_path, f'{id_}.txt'))
            img2 = cv2.cvtColor(cv2.imread(os.path.join(self.img2_path, f'{id_}.png')), cv2.COLOR_BGR2RGB)
            img3 = cv2.cvtColor(cv2.imread(os.path.join(self.img3_path, f'{id_}.png')), cv2.COLOR_BGR2RGB)
            objects_L = [Object3d(line) for line in open(label2_txt).readlines()]
            objects_R = [Object3d(line) for line in open(label3_txt).readlines()]
            # use left image obj-level as standard, or box-height results in differenet difficulty
            for obj_L, obj_R in zip(objects_L, objects_R):
                if obj_L.cls_type in self.cls_list and obj_L.level in self.diff_list:
                    obj_L.set_crop(img2, cam_to_img, 'left')
                    all_objects_L.append(obj_L)
                    obj_R.set_crop(img3, cam_to_img, 'right')
                    all_objects_R.append(obj_R)
                    self.update_cls_dims(obj_L.cls_type, obj_L.dim) # for calcualte dim avg
        return all_objects_L, all_objects_R

    def update_cls_dims(self, cls, dim):
        if cls not in self.cls_dims.keys():
            self.cls_dims[cls] = dict()
            self.cls_dims[cls]['dim_sum'] = np.array(dim, dtype=np.float32)
            self.cls_dims[cls]['count'] = 1
        else:
            self.cls_dims[cls]['dim_sum']+= np.array(dim, dtype=np.float32)
            self.cls_dims[cls]['count'] += 1
    
    def get_cls_dim_avg(self, cls):
        return self.cls_dims[cls]['dim_sum'] / self.cls_dims[cls]['count']
    
    def get_targets(self, objects_L, objects_R):
        targets_L = list()
        targets_R = list()
        for obj_L, obj_R in zip(objects_L, objects_R):
            # left image
            obj_target = dict()
            obj_target['Class'] = obj_L.cls_type
            obj_target['Truncation'] = obj_L.trucation
            obj_target['Box2d']: obj_L.box2d
            obj_target['Alpha'] = obj_L.alpha
            obj_target['Ry'] = obj_L.ry
            obj_target['Dim_delta']= obj_L.dim - self.get_cls_dim_avg(obj_L.cls_type)
            obj_target['Location']= obj_L.pos
            obj_target['Heading_bin'], obj_target['Heading_res'] = angle2class(obj_L.alpha, self.bins)
            obj_target['Theta_ray'] = obj_L.theta_ray
            targets_L.append(obj_target)
            # right image
            obj_target = dict()
            obj_target['Class'] = obj_R.cls_type
            obj_target['Truncation'] = obj_R.trucation
            obj_target['Box2d']: obj_R.box2d
            obj_target['Alpha'] = obj_R.alpha
            obj_target['Ry'] = obj_R.ry
            obj_target['Dim_delta']= obj_R.dim - self.get_cls_dim_avg(obj_R.cls_type)
            obj_target['Location']= obj_R.pos
            obj_target['Heading_bin'], obj_target['Heading_res'] = angle2class(obj_R.alpha, self.bins)
            obj_target['Theta_ray'] = obj_R.theta_ray
            targets_R.append(obj_target)
        return targets_L, targets_R

# modified from monodle
class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0].lower()
        self.trucation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        # str->float->np.int32
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.int32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.dim = np.array([self.h, self.w, self.l], dtype=np.float32)
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        #self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        #self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()
        self.crop = None
        self.calib = None
        self.camera_pose = None
        self.theta_ray = None
    
    def set_crop(self, img, calib, camera_pose):
        self.crop = img[self.box2d[1]:self.box2d[3]+1, self.box2d[0]:self.box2d[2]+1]
        self.calib = calib
        self.camera_pose = camera_pose.lower()
        self.theta_ray = self.calc_theta_ray(img.shape[1], self.box2d, calib, camera_pose)
    
    def calc_theta_ray(self, width, box2d, cam_to_img, camera_pose):#透過跟2d bounding box 中心算出射線角度
        if camera_pose == 'left':
            fovx = 2 * np.arctan(width / (2 * cam_to_img.p2[0][0]))
        elif camera_pose == 'right':
            fovx = 2 * np.arctan(width / (2 * cam_to_img.p3[0][0]))
        x_center = (box2d[0] + box2d[2]) // 2
        dx = x_center - (width // 2)
        mult = 1 if dx >=0 else -1
        dx = abs(dx)
        angle = mult * np.arctan( (2*dx*np.tan(fovx/2)) / width )
        return angle_correction(angle)
                                
    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = 'DontCare'
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                            % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                                self.pos, self.ry)
        return print_str

    def to_kitti_format_label(self):
        left, top, right, btm = self.box2d
        H, W, L = self.h, self.w, self.l
        X, Y, Z = self.pos
        print_str = '%s %.1f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, left, top, right, btm,
                        W, H, L, X, Y, Z, self.ry)
        return print_str
    
    def REG_result_to_kitti_format_label(self, reg_alpha, reg_dim, reg_pos):
        left, top, right, btm = self.box2d
        H, W, L = reg_dim
        X, Y, Z = reg_pos
        reg_ry = self.ry - self.alpha + reg_alpha
        print_str = '%s %.1f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                     % (self.cls_type, self.trucation, self.occlusion, reg_alpha, left, top, right, btm,
                        W, H, L, X, Y, Z, reg_ry)
        return print_str

#https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/blob/63c6ab98b7a5e36c7bcfdec4529804fc940ee900/lib/model/utils/kitti_utils.py#L97C5-L97C25
class FrameCalibrationData:
    '''Frame Calibration Holder
        p0-p3      Camera P matrix. Contains extrinsic 3x4    
                   and intrinsic parameters.
        r0_rect    Rectification matrix, required to transform points 3x3    
                   from velodyne to camera coordinate frame.
        tr_velodyne_to_cam0     Used to transform from velodyne to cam 3x4    
                                coordinate frame according to:
                                Point_Camera = P_cam * R0_rect *
                                                Tr_velo_to_cam *
                                                Point_Velodyne.
    '''

    def __init__(self, calib_path):
        self.calib_path = calib_path
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p2_2 = []
        self.p2_3 = []
        self.r0_rect = []
        self.t_cam2_cam0 = []
        self.tr_velodyne_to_cam0 = []
        self.set_info(calib_path)
        
    def set_info(self, calib_path):
        ''' 
        Reads in Calibration file from Kitti Dataset.
        
        Inputs:
        CALIB_PATH : Str PATH of the calibration file.
        
        Returns:
        frame_calibration_info : FrameCalibrationData
                                Contains a frame's full calibration data.
        ^ z        ^ z                                      ^ z         ^ z
        | cam2     | cam0                                   | cam3      | cam1
        |-----> x  |-----> x                                |-----> x   |-----> x

        '''
        data_file = open(calib_path, 'r')
        data_reader = csv.reader(data_file, delimiter=' ')
        data = []

        for row in data_reader:
            data.append(row)

        data_file.close()

        p_all = []

        for i in range(4):
            p = data[i]
            p = p[1:]
            p = [float(p[i]) for i in range(len(p))]
            p = np.reshape(p, (3, 4))
            p_all.append(p)

        # based on camera 0
        self.p0 = p_all[0]
        self.p1 = p_all[1]
        self.p2 = p_all[2]
        self.p3 = p_all[3]

        # based on camera 2
        self.p2_2 = np.copy(p_all[2]) 
        self.p2_2[0,3] = self.p2_2[0,3] - self.p2[0,3]

        self.p2_3 = np.copy(p_all[3]) 
        self.p2_3[0,3] = self.p2_3[0,3] - self.p2[0,3]

        self.t_cam2_cam0 = np.zeros(3)
        self.t_cam2_cam0[0] = (self.p2[0,3] - self.p0[0,3])/self.p2[0,0]

        # Read in rectification matrix
        tr_rect = data[4]
        tr_rect = tr_rect[1:]
        tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
        self.r0_rect = np.reshape(tr_rect, (3, 3))

        # Read in velodyne to cam matrix
        tr_v2c = data[5]
        tr_v2c = tr_v2c[1:]
        tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
        self.tr_velodyne_to_cam0 = np.reshape(tr_v2c, (3, 4))