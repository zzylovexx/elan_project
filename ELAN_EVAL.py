from library.ron_utils import *
import sys, os
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--result-path', "-R_PATH", required=True, help='folder of the predict result')
parser.add_argument('--data-path', "-D_PATH", required=True, help='folder of the elan dataset')

def evaluation(result_root, data_root):
    try:
        val_ids = [x.strip() for x in open(f'{data_root}/ImageSets/val.txt').readlines()]
    except:
        # temp way for 230808 dataset
        all_labels = glob.glob(f'{data_root}/renew_label/*.txt')
        val_ids = [name.split('/')[-1].split('.')[0] for name in all_labels]
        print('Dataset Elan_230808:', len(val_ids))
        

    dim_GT = list()
    dim_ELAN = list()
    depth_GT = list()
    depth_ELAN = list()
    alpha_GT = list()
    alpha_ELAN = list()

    for id_ in val_ids:
        gt_lines = [x.strip() for x in open(f'{data_root}/renew_label/{id_}.txt').readlines()]
        gt_objects = [TrackingObject(line) for line in gt_lines if line.split()[0].lower()=='car']
        for obj in gt_objects:
            dim_GT.append(obj.dims[0])
            depth_GT.append(obj.locs[0][2])
            alpha_GT.append(obj.alphas[0])
            
        pred_lines = [x.strip() for x in open(f'{result_root}/label_2/{id_}.txt').readlines()]
        pred_objects = [TrackingObject(line) for line in pred_lines if line.split()[0].lower()=='car']
        for obj in pred_objects:
            dim_ELAN.append(obj.dims[0])
            depth_ELAN.append(obj.locs[0][2])
            alpha_ELAN.append(obj.alphas[0])

    dim_GT = np.array(dim_GT)
    dim_ELAN = np.array(dim_ELAN)
    depth_GT = np.array(depth_GT)
    depth_ELAN = np.array(depth_ELAN)
    alpha_GT = np.array(alpha_GT)
    alpha_ELAN = np.array(alpha_ELAN)

    depth_diff = depth_GT-depth_ELAN
    alpha_diff = np.cos(alpha_GT - alpha_ELAN)
    dim_diff = np.mean(abs(dim_GT-dim_ELAN), axis=0)
    print(f'[Depth diff] abs_mean: {abs(depth_diff).mean():.4f}')
    print(f'[Alpha diff] abs_mean: {1-alpha_diff.mean():.4f}')
    print(f'[DIM diff] H:{dim_diff[0]:.4f}, W:{dim_diff[1]:.4f}, L:{dim_diff[2]:.4f}')
    print('[Depth error]')
    box_depth_error_calculation(depth_GT, depth_ELAN, 5)

def main():
    FLAGS = parser.parse_args()
    result_root = FLAGS.result_path
    data_root = FLAGS.data_path
    #make folder
    eval_folder = f'{data_root.replace("/","")}_eval/{result_root.split("/")[0]}'
    eval_txt = os.path.join(eval_folder, f'{result_root.split("/")[-1]}.txt')
    os.makedirs(eval_folder, exist_ok=True)
    org_stdout = sys.stdout
    f = open(eval_txt, 'w')
    sys.stdout = f
    evaluation(result_root, data_root)
    sys.stdout = org_stdout
    f.close()
    #evaluation(result_root, data_root)
    print(f'save in {eval_txt}')

if __name__ == '__main__':
    main()