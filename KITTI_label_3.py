import glob
import cv2
import os
import numpy as np
from library.Math import create_corners, rotation_matrix
from library.Plotting import project_3d_pt
from library.ron_utils import FrameCalibrationData

def generate_Kitti_label_3():
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
            occluded = int(elements[2])
            alpha = elements[3]
            left = int(round(elements[4]))
            top = int(round(elements[5]))
            right = int(round(elements[6]))
            btm = int(round(elements[7]))
            dim = [elements[8], elements[9], elements[10]]
            loc = [elements[11], elements[12], elements[13]]
            ry = elements[14]
            # calc box2d of image_3
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
        if i%500==0:
            print(i)

if __name__ == '__main__':
    print('START')
    generate_Kitti_label_3()
    print('DONE')