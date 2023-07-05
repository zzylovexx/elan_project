import glob
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--type', type=int, default=0, help='0:Kitti dataset, 1:Elan dataset')

def calc_class_dimension_ratio(labels, filename):
    CLASS_INFO = dict()
    for i in range(len(labels)):
        lines = [x.strip() for x in open(labels[i]).readlines()]
        for line in lines:
            line = line.split(' ')
            class_ = line[0]
            if class_ == 'DontCare':
                continue
            if class_ not in CLASS_INFO.keys(): # init class dict
                CLASS_INFO[class_] = dict()
                #CLASS_INFO[class_]['Dim'] = list()
                CLASS_INFO[class_]['L_H_ratio'] = 0
                CLASS_INFO[class_]['L_W_ratio'] = 0
                CLASS_INFO[class_]['L_HW_ratio'] = 0
                CLASS_INFO[class_]['count'] = 0
            dimension = [float(line[8]), float(line[9]), float(line[10])]
            L_H_ratio = dimension[2] / dimension[0]
            L_W_ratio = dimension[2] / dimension[1]
            L_HW_ratio = dimension[2] / dimension[1] / dimension[0]
            
            #CLASS_INFO[class_]['Dim'].append(dimension)
            CLASS_INFO[class_]['L_H_ratio'] += L_H_ratio
            CLASS_INFO[class_]['L_W_ratio'] += L_W_ratio
            CLASS_INFO[class_]['L_HW_ratio'] += L_HW_ratio
            CLASS_INFO[class_]['count'] += 1
    
    with open(filename, "w") as f:
        f.write(json.dumps(CLASS_INFO))
    return CLASS_INFO

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    kitti_labels = glob.glob('Kitti/training/label_2/*.txt')
    elan_labels = glob.glob('Elan_3d_box/label_2/*.txt')

    if FLAGS.type == 0: # Kitti
        print('Kitti')
        CLASS_INFO = calc_class_dimension_ratio(kitti_labels, 'kitti_dim_ratio.txt')
        print(CLASS_INFO)
    elif FLAGS.type == 1: # Elan
        print('Elan')
        CLASS_INFO = calc_class_dimension_ratio(elan_labels, 'elan_dim_ratio.txt')
        print(CLASS_INFO)
    else:
        print('WRONG TYPE VALUE')
