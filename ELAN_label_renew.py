import os
import glob

import numpy as np

def angle_correction(angle):
    if angle > np.pi:
        angle -= 2*np.pi
    elif angle < -1*np.pi:
        angle += 2*np.pi
    return angle

def check_box_range(left, right, top, btm, img_W=1280, img_H=720):
    if left < 1 or left > img_W:
        return False
    if right < 1 or right > img_W:
        return False
    if btm < 1 or btm > img_H:
        return False
    if top < 1 or top > img_H:
        return False
    return True

def box_correction(left, right, top, btm, img_W=1280, img_H=720):
    left = max(left, 1)
    right = min(right, img_W)
    top = max(top, 1)
    btm = min(btm, img_H)
    return left, right, top, btm

def main():
    #data_root = 'Elan_3d_box' 
    data_root = 'Elan_3d_box_230808'
    labels = glob.glob(f'{data_root}/label_2/*.txt')
    print(len(labels))
    img_W  = 1280
    img_H = 720
    os.makedirs(f'{data_root}/renew_label', exist_ok=True)
    for i in range(len(labels)):
        #ids = [int(x.strip()) for x in open(split_dir).readlines()]
        renew_labels = ''
        lines = [x.strip() for x in open(labels[i]).readlines()]
        for line in lines:
            elements = line.split()
            obj_class = elements[0]
            
            for j in range(1, len(elements)):
                elements[j] = float(elements[j])
            
            left = int(elements[4])
            top = int(elements[5])
            right = int(elements[6])
            btm = int(elements[7])

            left, right, top, btm = box_correction(left, right, top, btm, img_W, img_H)
            if abs(left-right) < 2 or abs(top-btm) < 2:
                continue
            # 2d box out of image
            elif check_box_range(left, right, top, btm, img_W, img_H):
            #correct labels!
                truncated = elements[1]
                occluded = elements[2]
                alpha = float(elements[3])
                alpha = angle_correction(alpha)
                ry = float(elements[14])
                ry = angle_correction(ry)

                dim = [elements[8], elements[9], elements[10]]
                loc = [elements[11], elements[12], elements[13]]

                renew_labels += '{CLASS} {T:.1f} {O} {A:.2f} {left} {top} {right} {btm} {H:.2f} {W:.2f} {L:.2f} {X:.2f} {Y:.2f} {Z:.2f} {Ry:.2f}\n'.format(
                            CLASS=obj_class, T=truncated, O=occluded, A=alpha, left=left, top=top, right=right, btm=btm,
                            H=dim[0], W=dim[1], L=dim[2], X=loc[0], Y=loc[1], Z=loc[2], Ry=ry)
        
        with open(labels[i].replace('label_2','renew_label'), 'w') as new_f:
            new_f.writelines(renew_labels)
        
        if i%500 == 0:
            print(i)

if __name__=='__main__':
    main()