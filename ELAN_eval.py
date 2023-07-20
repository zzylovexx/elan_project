from library.ron_utils import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--result-path', required=True, help='folder of the predict result')

def evaluation(result_root):
    valset = [x.strip() for x in open('Elan_3d_box/ImageSets/val.txt').readlines()]
    dim_GT = list()
    dim_ELAN = list()
    depth_GT = list()
    depth_ELAN = list()
    alpha_GT = list()
    alpha_ELAN = list()

    for id_ in valset:
        gt_lines = [x.strip() for x in open(f'Elan_3d_box/renew_label/{id_}.txt').readlines()]
        gt_objects = [TrackingObject(line) for line in gt_lines if line.split()[0]=='Car']
        for obj in gt_objects:
            dim_GT.append(obj.dims[0])
            depth_GT.append(obj.locs[0][2])
            alpha_GT.append(obj.alphas[0])
            
        pred_lines = [x.strip() for x in open(f'{result_root}/label_2/{id_}.txt').readlines()]
        pred_objects = [TrackingObject(line) for line in pred_lines if line.split()[0]=='Car']
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
    alpha_diff = alpha_GT - alpha_ELAN
    dim_diff = np.mean(abs(dim_GT-dim_ELAN), axis=0)
    print(f'[Depth diff] abs_mean: {abs(depth_diff).mean():.4f}')
    print(f'[Alpha diff] abs_mean: {abs(alpha_diff).mean():.4f}')
    print(f'[DIM diff] H:{dim_diff[0]:.4f}, W:{dim_diff[1]:.4f}, L:{dim_diff[2]:.4f}')
    print('[Depth error]')
    box_depth_error_calculation(depth_GT, depth_ELAN, 5)

def main():
    FLAGS = parser.parse_args()
    result_root = FLAGS.result_path
    evaluation(result_root)

if __name__ == '__main__':
    main()