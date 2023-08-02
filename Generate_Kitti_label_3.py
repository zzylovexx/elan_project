
import csv


import glob
import cv2
import os
import numpy as np
from library.Math import create_corners, rotation_matrix
from library.Plotting import project_3d_pt, plot_3d_box

def main():
    calib = sorted(glob.glob('Kitti/training/calib/*.txt'))
    images_2 = sorted(glob.glob('Kitti/training/image_2/*.png'))
    labels_2 = sorted(glob.glob('Kitti/training/label_2/*.txt'))
    os.makedirs('Kitti/training/label_3', exist_ok=True)
    for i in range(len(images_2)):
        lines = [x.strip() for x in open(labels_2[i]).readlines()]
        img = cv2.cvtColor(cv2.imread(images_2[i]), cv2.COLOR_BGR2RGB)
        cam_to_img = FrameCalibrationData(calib[i])
        H, W, _ = img.shape
        label_3 = ''
        for line in lines:
            #LABEL_2
            elements = line.split()
            class_ = elements[0]
            if elements[0] == 'DontCare':
                continue
            for j in range(1,len(elements)):
                elements[j] = float(elements[j])
            truncated = elements[1]
            occluded = elements[2]
            alpha = elements[3]
            left = int(round(elements[4]))
            top = int(round(elements[5]))
            right = int(round(elements[6]))
            btm = int(round(elements[7]))
            dim = [elements[8], elements[9], elements[10]]
            loc = [elements[11], elements[12], elements[13]]
            ry = elements[14]
            # calc box_2d of image_3
            X = list()
            Y = list()
            R = rotation_matrix(ry)
            #https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/blob/63c6ab98b7a5e36c7bcfdec4529804fc940ee900/lib/model/utils/kitti_utils.py#L195C68-L195C86
            loc_correction = loc + cam_to_img.t_cam2_cam0 #3d location is for cam0, need to add relative cam2_cam0 distance
            loc_correction[1] -= dim[0]/2 # move to center
            corners_3d = create_corners(dim, loc_correction, R)
            for pt_3d in corners_3d:
                pt_2d = project_3d_pt(pt_3d, cam_to_img.p2_3) #proj by calib_2_3
                X.append(pt_2d[0])
                Y.append(pt_2d[1])
            left = max(min(X),0)
            right = min(max(X), W-1)
            top = max(min(Y), 0)
            btm = min(max(Y), H-1)
            
            label_3 += '{CLASS} {T:.1f} {O} {A:.2f} {left} {top} {right} {btm} {H:.2f} {W:.2f} {L:.2f} {X:.2f} {Y:.2f} {Z:.2f} {Ry:.2f}\n'.format(
                        CLASS=class_, T=truncated, O=occluded, A=alpha, left=left, top=top, right=right, btm=btm,
                        H=dim[0], W=dim[1], L=dim[2], X=loc[0], Y=loc[1], Z=loc[2], Ry=ry)
            
        with open(labels_2[i].replace('label_2', 'label_3'), 'w') as f:
            f.writelines(label_3)
    print('DONE')

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

if __name__ == '__main__':
    main()