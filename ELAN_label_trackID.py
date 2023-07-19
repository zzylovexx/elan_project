import os
import glob
import numpy as np
import cv2
from library.ron_utils import *

def tracking_obj_by_labels(labels, start, end, WRITE_FILE=False, folder='renew_label', new_folder='renew_label_obj'):
    tracking_dict = dict()
    for idx in range(start, end):
        #print(f'======{idx}====={label[idx]}====')
        lines = [x.strip() for x in open(labels[idx]).readlines()]
        new_lines = [line for line in lines]
        #img = cv2.cvtColor(cv2.imread(images[idx]), cv2.COLOR_BGR2RGB)
        objects = [DetectedObject(line) for line in lines]
        if len(tracking_dict.keys()) == 0:
            for id_, obj in enumerate(objects):
                
                tracking_dict[id_] = obj
                tracking_dict[id_].record_frames(idx)
                new_lines[id_] += f' {id_}'
                #top_left, btm_right = obj.box2d
                #crop = img[top_left[1]:btm_right[1]+1, top_left[0]:btm_right[0]+1]
                #tracking_dict[id_].crops.append(crop)
        else:
            for obj_idx, obj in enumerate(objects):
                #print(obj.class_)
                now_box2d = obj.box2d
                match = False
                #top_left, btm_right = obj.box2d
                #crop = img[top_left[1]:btm_right[1]+1, top_left[0]:btm_right[0]+1]
                for key in tracking_dict.keys():
                    last_box2d = tracking_dict[key].box2d
                    iou_value = iou_2d(now_box2d, last_box2d)
                    last_frame = tracking_dict[key].frames[-1]
                    if iou_value > 0.6 and idx - last_frame < 5:
                        #print(f'MATCHED:{iou_value:.2f}')
                        match = True
                        tracking_dict[key].update_info(obj)
                        tracking_dict[key].record_frames(idx)
                        #tracking_dict[key].crops.append(crop)
                        new_lines[obj_idx] += f' {key}'
                        break
                if not match: #new obj
                    new_id = len(tracking_dict.keys())
                    tracking_dict[new_id] = obj
                    tracking_dict[new_id].record_frames(idx)
                    #tracking_dict[new_id].crops.append(crop)
                    new_lines[obj_idx] += f' {key}'
        if WRITE_FILE:
            with open(labels[idx].replace(folder, new_folder), 'w') as f:
                for line in new_lines:
                    f.writelines(line + '\n')
        if idx%500 == 0:
            print(idx)
    return tracking_dict

def main():
    renew_labels = glob.glob('Elan_3d_box/renew_label/*.txt')
    os.makedirs('Elan_3d_box/renew_label_obj', exist_ok=True)
    tracking_dict = tracking_obj_by_labels(renew_labels, 0, len(renew_labels), WRITE_FILE=True)

if __name__=='__main__':
    main()