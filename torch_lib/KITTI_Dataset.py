import os, cv2, csv
from torch.utils import data
import numpy as np
from library.ron_utils import angle2class, angle_correction, flip_orient, FrameCalibrationData
from KITTI_label_3 import generate_Kitti_label_3

class KITTI_Dataset(data.Dataset):
    def __init__(self, cfg, process, split='train', is_flip=False):
        path = cfg['path']
        self.label2_path = os.path.join(path, 'label_2')
        self.img2_path = os.path.join(path, 'image_2')
        if os.name.lower()=='posix':
            print('Load Right Images')
            self.label3_path = os.path.join(path, 'label_3')
            #TODO label3
            if not os.path.isdir(self.label3_path): # generate label_3
                print('Generating Kitti_label_3')
                generate_Kitti_label_3()
            self.img3_path = os.path.join(path, 'image_3')
        elif os.name.lower()=='nt':
            self.label3_path = os.path.join(path, 'label_2')
            self.img3_path = os.path.join(path, 'image_2')

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
        self.is_flip = is_flip #horizontal
        self.objects_L, self.objects_R = self.get_objects(self.ids)
        self.targets_L, self.targets_R = self.get_targets(self.objects_L, self.objects_R)
        self.transform = process
        
    def __len__(self):
        return len(self.objects_L)

    def __getitem__(self, idx):
        # left_obj, left_label, right_obj, right_label
        return self.transform(self.objects_L[idx].crop), self.targets_L[idx],  \
                self.transform(self.objects_R[idx].crop), self.targets_R[idx]

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
                    obj_L.set_crop(img2, cam_to_img, 'left', self.is_flip)
                    all_objects_L.append(obj_L)
                    obj_R.set_crop(img3, cam_to_img, 'right', self.is_flip)
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
            obj_target['Truncation'] = obj_L.truncation
            obj_target['Box2d'] = obj_L.box2d
            obj_target['Alpha'] = obj_L.alpha
            obj_target['Ry'] = obj_L.ry
            obj_target['Dim_delta'] = obj_L.dim - self.get_cls_dim_avg(obj_L.cls_type)
            obj_target['Location'] = obj_L.pos
            obj_target['Depth'] = obj_L.pos[2]
            obj_target['Heading_bin'], obj_target['Heading_res'] = angle2class(obj_L.alpha, self.bins)
            obj_target['Theta_ray'] = obj_L.theta_ray
            obj_target['Calib'] = obj_L.calib.p2
            obj_target['img_W'] = obj_L.img_W
            targets_L.append(obj_target)
            # right image
            obj_target = dict()
            obj_target['Class'] = obj_R.cls_type
            obj_target['Truncation'] = obj_R.truncation
            obj_target['Box2d'] = obj_R.box2d
            obj_target['Alpha'] = obj_R.alpha
            obj_target['Ry'] = obj_R.ry
            obj_target['Dim_delta']= obj_R.dim - self.get_cls_dim_avg(obj_R.cls_type)
            obj_target['Location'] = obj_R.pos
            obj_target['Heading_bin'], obj_target['Heading_res'] = angle2class(obj_R.alpha, self.bins)
            obj_target['Theta_ray'] = obj_R.theta_ray
            obj_target['Calib'] = obj_L.calib.p3
            obj_target['img_W'] = obj_L.img_W
            targets_R.append(obj_target)
        return targets_L, targets_R

# modified from monodle
class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0].lower()
        self.truncation = float(label[1])
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
        self.img_W = None
        self.crop = None
        self.calib = None
        self.camera_pose = None
        self.theta_ray = None
        self.frame = None
        self.track_id = None
    
    def set_track_info(self, frame, track_id):
        self.frame = frame
        self.track_id = track_id

    def set_crop(self, img, calib, camera_pose, is_flip=False):
        self.img_W = img.shape[1]
        self.crop = img[self.box2d[1]:self.box2d[3]+1, self.box2d[0]:self.box2d[2]+1]
        if is_flip: #horizontal flip
            self.crop = cv2.flip(self.crop, 1)
            self.alpha = flip_orient(self.alpha)
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

        if self.truncation == -1:
            self.level_str = 'DontCare'
            return 0

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
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
                            % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                                self.pos, self.ry)
        return print_str

    def to_kitti_format_label(self):
        left, top, right, btm = self.box2d
        H, W, L = self.h, self.w, self.l
        X, Y, Z = self.pos
        print_str = '%s %.1f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, left, top, right, btm,
                        W, H, L, X, Y, Z, self.ry)
        return print_str
    
    def REG_result_to_kitti_format_label(self, **reg_values): #keys:alpha, dim, pos, trun
        left, top, right, btm = self.box2d
        reg_alpha = reg_values['alpha']
        reg_dim = reg_values['dim']
        reg_pos = reg_values['pos']
        reg_trun = reg_values['trun'] if 'trun' in reg_values.keys() else self.truncation
        H, W, L = reg_dim
        X, Y, Z = reg_pos
        reg_ry = self.ry - self.alpha + reg_alpha
        print_str = '%s %.1f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                     % (self.cls_type, reg_trun, self.occlusion, reg_alpha, left, top, right, btm,
                        W, H, L, X, Y, Z, reg_ry)
        return print_str