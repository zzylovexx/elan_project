import os, glob, cv2, time
import numpy as np
from library.ron_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', '-T', type=int, default=0, help='0:kitti, 1:ELAN')

def main():
    FLAGS = parser.parse_args()
    if FLAGS.type==0:
        print('Kitti dataset')
        images = glob.glob('Kitti/training/image_2/*.png')
        labels = glob.glob('Kitti/training/label_2/*.txt')
        label_folder = 'label_2'
        calibs = glob.glob('Kitti/training/calib/*.txt')
        os.makedirs('Kitti/training/extra_label', exist_ok=True)
    elif FLAGS.type==1:
        print('ELAN dataset')
        images = glob.glob('Elan_3d_box/image_2/*.png')
        labels = glob.glob('Elan_3d_box/renew_label/*.txt')
        label_folder = 'renew_label'
        cam_to_img = np.array([
        [ 1.418667e+03, 0.000000e+00, 6.4e+02,0],
        [ 0.000000e+00, 1.418667e+03, 3.6e+02,0],
        [ 0.000000e+00, 0.000000e+00, 1.000000e+00,0]])
        os.makedirs('Elan_3d_box/extra_label', exist_ok=True)
    
    #alpha_diff_txt = list()
    #ry_diff_txt = list()
    print('Start labeling')
    start = time.time()
    count = 0
    for i in range(len(images)):

        img = cv2.imread(images[i])
        img_width = img.shape[1]
        if FLAGS.type==0:
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

        alpha_classes = get_bin_classes(alphas, num_bin=60)
        ry_classes = get_bin_classes(rys, num_bin=60)
        for W, H, X, Y, T, alpha_G, ry_G in zip(box_width, box_height, offsetX, offsetY, theta, alpha_classes, ry_classes):
            new_label = str(img_width)+' '+str(W)+' '+str(H)+' '+str(X)+' '+str(Y)+' '+str(T)+' '
            new_label += str(i*100+alpha_G)+' '+str(i*100+ry_G) # seperate different images
            new_labels.append(new_label)

        # write new .txt
        with open(labels[i].replace(label_folder, 'extra_label'), 'w') as new_f:
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

    print('Done, take {} min {:.2f} sec'.format((time.time()-start)//60, (time.time()-start)%60))

if __name__=='__main__':
    main()