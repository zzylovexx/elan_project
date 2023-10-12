import os, cv2, glob
from torch.utils import data
import numpy as np
from library.ron_utils import angle2class, angle_correction, flip_orient
from ELAN_label_renew import label_renew

class ELAN_Dataset(data.Dataset):
    def __init__(self, cfg, process, is_flip=False, img_extension='png'):
        path = cfg['path']
        if not os.path.isdir(os.path.join(path, 'renew_label')): #label_2 with wrong objects
            label_renew(path)
        self.label2_path = os.path.join(path, 'renew_label')
    
        self.img2_path = os.path.join(path, 'image_2')
        #self.extra_label_path = os.path.join(path, 'extra_label') #using generated extra label
        self.cam_to_img = np.array([
            [1.418667e+03, 0.000e+00, 6.4e+02, 0],
            [0.000e+00, 1.418867e+03, 3.6e+02, 0],
            [0.000e+00, 000e+00, 1.0e+00, 0] ])

        self.transform = process
        self.ext = img_extension
        self.bins = cfg['bins']
        self.diff_list = cfg['diff_list']
        self.cls_list = cfg['class_list']
        self.cond = cfg['cond']
        self.ids = self.get_ids(self.label2_path)
        self.cls_dims = dict()
        self.is_flip = is_flip #horizontal
        self.objects = self.get_objects(self.ids)
        self.targets = self.get_targets(self.objects)
        
        
    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        # left_obj, left_label, right_obj, right_label
        try:
            return self.transform(self.objects[idx].crop), self.targets[idx]
        except:
            print(idx)
            return self.transform(self.objects[idx].crop), self.targets[idx]
    
    def get_ids(self, label_path):
        all_labels = sorted(glob.glob(f'{label_path}/*.txt'))
        #https://shengyu7697.github.io/python-detect-os/
        if os.name.lower() == 'posix': #ubuntu
            all_ids = [name.split('/')[-1].split('.')[0] for name in all_labels]
        elif os.name.lower() == 'nt': #windows
            all_ids = [name.split('\\')[-1].split('.')[0] for name in all_labels] 
        return all_ids

    def get_objects(self, ids):
        all_objects = list()
        for id_ in ids:
            label2_txt = os.path.join(self.label2_path, f'{id_}.txt')
            img2 = cv2.cvtColor(cv2.imread(os.path.join(self.img2_path, f'{id_}.{self.ext}')), cv2.COLOR_BGR2RGB)
            objects = [Object3d(line) for line in open(label2_txt).readlines()]
    
            # use left image obj-level as standard, or box-height results in differenet difficulty
            for obj in objects:
                if obj.cls_type in self.cls_list and obj.level in self.diff_list:
                    obj.set_crop(img2, self.cam_to_img, self.is_flip)
                    all_objects.append(obj)
                    self.update_cls_dims(obj.cls_type, obj.dim) # for calcualte dim avg
        return all_objects

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
    
    def get_targets(self, objects):
        targets = list()
        for obj in objects:
            # left image
            obj_target = dict()
            obj_target['Class'] = obj.cls_type
            obj_target['Truncation'] = obj.truncation
            obj_target['Box2d'] = obj.box2d
            obj_target['Alpha'] = obj.alpha
            obj_target['Ry'] = obj.ry
            obj_target['Dim_delta'] = obj.dim - self.get_cls_dim_avg(obj.cls_type)
            obj_target['Location'] = obj.pos
            obj_target['Depth'] = obj.pos[2]
            obj_target['Heading_bin'], obj_target['Heading_res'] = angle2class(obj.alpha, self.bins)
            obj_target['Theta_ray'] = obj.theta_ray
            obj_target['Calib'] = self.cam_to_img
            obj_target['img_W'] = obj.img_W
            targets.append(obj_target)
        return targets

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
        self.theta_ray = None
        self.frame = None
        self.track_id = None
    
    def set_track_info(self, frame, track_id):
        self.frame = frame
        self.track_id = track_id

    def set_crop(self, img, cam_to_img, is_flip=False):
        self.img_W = img.shape[1]
        self.crop = img[self.box2d[1]:self.box2d[3]+1, self.box2d[0]:self.box2d[2]+1]
        if is_flip: #horizontal flip
            self.crop = cv2.flip(self.crop, 1)
            self.alpha = flip_orient(self.alpha)
        self.theta_ray = self.calc_theta_ray(img.shape[1], self.box2d, cam_to_img)
    
    def calc_theta_ray(self, width, box2d, cam_to_img):#透過跟2d bounding box 中心算出射線角度
        fovx = 2 * np.arctan(width / (2 * cam_to_img[0][0]))
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
    
    def REG_result_to_kitti_format_label(self, reg_alpha, reg_dim, reg_pos):
        left, top, right, btm = self.box2d
        H, W, L = reg_dim
        X, Y, Z = reg_pos
        reg_ry = self.ry - self.alpha + reg_alpha
        print_str = '%s %.1f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                     % (self.cls_type, self.truncation, self.occlusion, reg_alpha, left, top, right, btm,
                        W, H, L, X, Y, Z, reg_ry)
        return print_str