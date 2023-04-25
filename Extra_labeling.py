import os, glob, cv2, time
import numpy as np
import matplotlib.pyplot as plt
from library.Plotting import *
from library.ron_utils import *
from library.File import *
from Alpha_Ry_labeling import *

def extra_labeling(img_root='Kitti/training/image_2', 
                   label_root='Kitti/training/label_2', 
                   calib_root='Kitti/training/calib',
                   extra_label_root = "Kitti/training/extra_label"):
    img_root = "./Kitti/training/image_2"
    label_root = "./Kitti/training/label_2"
    calib_root = "./Kitti/training/calib"
    images = glob.glob(os.path.join(img_root, '*.png'), recursive=True)
    labels = glob.glob(os.path.join(label_root, '*.txt'), recursive=True)
    calibs = glob.glob(os.path.join(calib_root, '*.txt'), recursive=True)
    extra_label_root = "./Kitti/training/extra_label"
    os.makedirs(extra_label_root, exist_ok=True)

    #alpha_diff_txt = list()
    #ry_diff_txt = list()
    print('Start labeling')
    start = time.time()
    count = 0
    for i in range(len(images)):

        img = cv2.imread(images[i])
        img_width = img.shape[1] 
        cam_to_img = get_calibration_cam_to_image(calibs[i])

        new_labels = list()

        box_width = list()
        box_height = list()
        offsetX = list()
        offsetY = list()
        theta = list()
        alphas = list()
        rys = list()

        with open(labels[i]) as f:

            lines = f.readlines()

            for line in lines:
                elements = line[:-1].split()
                if elements[0] == 'DontCare':
                    continue

                for j in range(1, len(elements)):
                    elements[j] = float(elements[j])

                # box_width, box_height
                top_left = (int(round(elements[4])), int(round(elements[5])))
                btm_right = (int(round(elements[6])), int(round(elements[7])))
                box = [top_left, btm_right]
                width, height = get_box_size(box)
                box_width.append(width)
                box_height.append(height)

                # offsetX, offsetY
                Dimension = np.array([elements[8], elements[9], elements[10]], dtype=np.double) # height, width, length
                d3_location = [elements[11], elements[12], elements[13]] # x, y, z
                d3_location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object
                offset_x, offset_y = calc_offset(box, d3_location, cam_to_img) #OK
                offsetX.append(offset_x)
                offsetY.append(offset_y)

                #theta
                theta_ray = round(calc_theta_ray(img_width, box, cam_to_img), 3)
                theta.append(theta_ray)

                #alpha, rotation_y
                alphas.append(elements[3])
                rys.append(elements[14])

        alpha_classes, alpha_diff = get_bin_classes(alphas, num_bin=60)
        ry_classes, ry_diff = get_bin_classes(rys, num_bin=60)
        for W, H, X, Y, T, alpha_G, ry_G in zip(box_width, box_height, offsetX, offsetY, theta, alpha_classes, ry_classes):
            new_label = str(img_width)+' '+str(W)+' '+str(H)+' '+str(X)+' '+str(Y)+' '+str(T)+' '
            new_label += str(i*100+alpha_G)+' '+str(i*100+ry_G) # seperate different images
            new_labels.append(new_label)

        # write new .txt
        with open(labels[i].replace(label_root, extra_label_root), 'w') as new_f:
            for label in new_labels:
                new_f.writelines(label+'\n')

        if count%500 == 0:
            print(count)
        count+=1

    #record diff(org!=new) txts
    '''
        if alpha_diff:
            alpha_diff_txt.append(files[i])
        if ry_diff:
            ry_diff_txt.append(files[i])

    with open('Kitti/training/alpha_diff.txt', 'w') as f:
        for txt in alpha_diff_txt:
            f.writelines(txt+'\n')
    with open('Kitti/training/ry_diff.txt', 'w') as f:
        for txt in ry_diff_txt:
            f.writelines(txt+'\n')
    '''

    print('Done, take {} min {} sec'.format((time.time()-start)//60, (time.time()-start)%60))
    print(f'Saved in {extra_label_root}')

if __name__=='__main__':
    extra_labeling()