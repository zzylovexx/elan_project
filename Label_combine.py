import glob, os
import numpy as np
import argparse 

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--zzy', "-Z", required=True, help='folder of zzys label')
parser.add_argument("--ron", "-R", required=True, help='folder of rons label')

def main():
    FLAGS = parser.parse_args()
    zzys_folder = FLAGS.zzy
    rons_folder = FLAGS.ron
    zzys_labels = sorted(glob.glob(f'{zzys_folder}/label_2/*.txt'))
    rons_labels = sorted(glob.glob(f'{rons_folder}/label_2/*.txt'))
    os.makedirs(f'{zzys_folder}/COMBINE_RESULT', exist_ok=True)
    for i in range(len(zzys_labels)):
        zzys_objects = [Object3d(line) for line in open(zzys_labels[i]).readlines()]
        rons_objects = [Object3d(line) for line in open(rons_labels[i]).readlines()]
        label_combine = ''
        for j in range(len(rons_objects)):
            match_value = 0
            obj_ron = rons_objects[j]
            for k in range(len(zzys_objects)):
                iou_value = calc_IoU_2d(obj_ron.box2d, zzys_objects[k].box2d)
                if iou_value > match_value:
                    match_value = iou_value
                    match_idx = k
            obj_zzy = zzys_objects[match_idx]

            alpha_ron = obj_ron.alpha
            ry_ron = obj_ron.ry
            gt_box2d = obj_ron.box2d
            class_ = obj_ron.cls_type
            truncation = obj_ron.truncation
            occlusion = obj_ron.occlusion
            
            dim_zzy = obj_zzy.dim
            loc_zzy = obj_zzy.pos

            label_combine += '{CLASS} {T:.1f} {O} {A:.2f} {left} {top} {right} {btm} {H:.2f} {W:.2f} {L:.2f} {X:.2f} {Y:.2f} {Z:.2f} {Ry:.2f}\n'.format(
                    CLASS=class_, T=truncation, O=occlusion, A=alpha_ron, left=gt_box2d[0], top=gt_box2d[1], right=gt_box2d[2], btm=gt_box2d[3],
                    H=dim_zzy[0], W=dim_zzy[1], L=dim_zzy[2], X=loc_zzy[0], Y=loc_zzy[1], Z=loc_zzy[2], Ry=ry_ron)

        new_label_path = zzys_labels[i].replace('label_2', 'COMBINE_RESULT')
        #print(new_label_path)
        with open(new_label_path, 'w') as f:
            f.writelines(label_combine)
    print('Save path:', new_label_path)

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

def calc_IoU_2d(box1, box2):
    box1 = np.array(box1, dtype=np.int32).flatten()
    box2 = np.array(box2, dtype=np.int32).flatten()
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    area_sum = abs(area1) + abs(area2)
    
    #計算重疊方形座標
    x1 = max(box1[0], box2[0]) # left
    y1 = max(box1[1], box2[1]) # top
    x2 = min(box1[2], box2[2]) # right
    y2 = min(box1[3], box2[3]) # btm

    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        area_overlap = abs((x2-x1)*(y2-y1))

    area_union = area_sum-area_overlap
    return area_overlap/area_union

if __name__=='__main__':
    main()
