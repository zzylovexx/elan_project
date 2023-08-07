import os, cv2
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
    def __init__(self, cfg, split='train', camera_pose='left'):
        path = cfg['path']
        self.camera_pose = camera_pose.lower()
        if self.camera_pose == 'left':
            self.label_path = os.path.join(path, 'label_2')
            self.img_path = os.path.join(path, 'image_2')
        elif self.camera_pose == 'right':
            self.label_path = os.path.join(path, 'label_3')
            self.img_path = os.path.join(path, 'image_3')
            
        self.calib_path = os.path.join(path, 'calib') 
        self.extra_label_path = os.path.join(path, 'extra_label') #using generated extra label
        self.bins = cfg['bins']
        self.diff_list = cfg['diff_list']
        self.cls_list = cfg['class_list']
        self.cond = cfg['cond']
        self.split = os.path.join('Kitti/ImageSets', split + '.txt')
        self.ids = [x.strip() for x in open(self.split).readlines()]
        self.cls_dims = dict()
        self.objects = self.get_objects(self.ids)
        self.targets = self.get_targets(self.objects)
    
    def get_objects(self, ids):
        all_objects = list()
        for id_ in ids:
            label_txt = os.path.join(self.label_path, f'{id_}.txt')
            cam_to_img = FrameCalibrationData(self.calib_path, f'{id_}.txt')
            img = cv2.cvtColor(cv2.imread(os.path.join(self.img_path, f'{id_}.png')), cv2.COLOR_BGR2RGB)
            objects = [Object3d(line, img, cam_to_img, self.camera_pose) for line in open(label_txt).readlines()]
            for obj in objects:
                if obj.cls_type in self.cls_list and obj.level in self.diff_list:
                    all_objects.append(obj)
                    self.update_cls_dims(obj.cls_type, obj.dim)
        return objects

    def update_cls_dims(self, cls, dim):
        if cls not in self.cls_dims.keys():
            self.cls_dims[cls] = list()
        self.cls_dims[cls].append(dim)
    
    def get_cls_dim_avg(self, cls):
        return np.mean(self.cls_dims[cls], axis=0)
    
    def get_targets(self, objects):
        targets = list()
        for obj in objects:
            obj_target = dict()
            obj_target['Class'] = obj.cls_type
            obj_target['Truncation'] = obj.trucation
            obj_target['Box2D']: obj.box2d
            obj_target['Alpha'] = obj.alpha
            obj_target['Ry'] = obj.ry
            obj_target['Dim_delta']: obj.dim - self.get_cls_dim_avg(obj.cls_type)
            obj_target['Location']: obj.pos
            obj_target['Heading_bin'], obj_target['Heading_res'] = angle2class(obj.alpha, self.bins)
            obj_target['Theta_ray'] = obj.theta_ray
            targets.append(obj_target)
        return targets

# modified from monodle
class Object3d(object):
    def __init__(self, line, img, calib, camera_pose):
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
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()
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