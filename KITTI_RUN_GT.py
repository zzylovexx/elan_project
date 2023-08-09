from torchvision.models import vgg
from torch_lib.Model_heading_bin import *
from torchvision import transforms
from torch_lib.KITTI_Dataset_0808 import *
from KITTI_EVAL import ron_evaluation
from library.ron_utils import *
import os, cv2, time, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", '-D', type=int, default=0, help='select cuda index')
# path setting
parser.add_argument("--weights-path", '-W_PATH', required=True, help='weighs path')
parser.add_argument("--result-path", '-R_PATH', required=True, help='path (folder name) of the generated pred-labels')

def main():
    #weights_path = 'weights/0808car/KITTI_BL_B4_50.pkl'
    #result_root = '0808car/KITTI_BL_B4_50'
    #os.makedirs(result_root, exist_ok=True)
    #device = torch.device('cuda:0') # 選gpu的index

    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_path
    result_root = FLAGS.result_path
    os.makedirs(result_root, exist_ok=True)

    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    checkpoint = torch.load(weights_path, map_location=device) #if training on 2 GPU, mapping on the same device
    bin_num = checkpoint['bin'] 
    is_cond = checkpoint['cond']
    #cfg = checkpoint['cfg']
    #bin_num = cfg['bins'] 
    #is_cond = cfg['cond']
    cfg = {'path':'Kitti/training',
            'class_list':['car'], 'diff_list': [1,2], #0:DontCare, 1:Easy, 2:Moderate, 3:Hard, 4:Unknown
            'bins': 4, 'cond':False}
    diff_list = cfg['diff_list']
    cls_list = cfg['class_list']
    angle_per_class = 2*np.pi/float(bin_num)

    my_vgg = vgg.vgg19_bn(weights='DEFAULT')
    if is_cond:
        print("< add Condition (4-dim) as input >")
        my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
    model = Model(features=my_vgg.features, bins=bin_num).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # for img processing
    process = transforms.Compose([transforms.ToTensor(), 
                                  transforms.Resize([224,224], transforms.InterpolationMode.BICUBIC), 
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    dataset_train = KITTI_Dataset(cfg, process, split='train')    
    img2_path = 'Kitti/training/image_2'
    label2_path = 'Kitti/training/label_2'
    calib_path = 'Kitti/training/calib'
    split = 'trainval'
    split_dir = f'Kitti/ImageSets/{split}.txt'
    ids = [x.strip() for x in open(split_dir).readlines()]

    for id_ in ids:
        label2_txt = os.path.join(label2_path, f'{id_}.txt')
        cam_to_img = FrameCalibrationData(os.path.join(calib_path, f'{id_}.txt'))
        img2 = cv2.cvtColor(cv2.imread(os.path.join(img2_path, f'{id_}.png')), cv2.COLOR_BGR2RGB)
        objects = [Object3d(line) for line in open(label2_txt).readlines()]
        inputs = list()
        REG_objects = list()
        for obj in objects:
            if obj.cls_type in cls_list and obj.level in diff_list:
                obj.set_crop(img2, cam_to_img, 'left')
                inputs.append(process(obj.crop))
                REG_objects.append(obj)
        reg_labels = ''
        if len(inputs)!=0:
            inputs = torch.stack(inputs).to(device)
            # model regress part
            [residual, bin_, dim] = model(inputs)
            bin_argmax = torch.max(bin_, dim=1)[1]
            orient_residual = residual[torch.arange(len(residual)), bin_argmax].detach()
            REG_alphas = angle_per_class*bin_argmax + orient_residual #mapping bin_class and residual to get alpha
        
            for i in range(len(inputs)):
                obj = REG_objects[i]
                reg_alpha = angle_correction(REG_alphas[i].detach().item())
                avg_dim = np.array(dataset_train.get_cls_dim_avg(obj.cls_type))
                reg_dim = avg_dim + dim[i].cpu().detach().numpy()
                reg_pos, _ = calc_location(reg_dim, cam_to_img.p2, obj.box2d.reshape((2,2)), reg_alpha, obj.theta_ray)
                reg_pos[1] += reg_dim[0]/2 #reg_pos is 3d center, + H/2 to compare with gt label
                #gt_label = obj.to_kitti_format_label()
                reg_labels += obj.REG_result_to_kitti_format_label(reg_alpha, reg_dim, reg_pos) + '\n'
                #print(gt_label)
                #print(reg_label)
                #gt_calc_depth = calc_depth_with_alpha_theta(img2.shape[1], obj.box2d, cam_to_img.p2, obj.w, obj.l, obj.alpha)
                #reg_calc_depth = calc_depth_with_alpha_theta(img2.shape[1], obj.box2d, cam_to_img.p2, reg_dim[1], reg_dim[2], reg_alpha)
                #depth_calc = calc_depth_with_alpha_theta(img_W, box2d, cam_to_img, obj_W, obj_L, alpha, trun)
                #print(reg_calc_depth, gt_calc_depth)
        with open(os.path.join(result_root, f'{id_}.txt'), 'w') as f:
            f.writelines(reg_labels)
    
    # eval part
    val_dir = 'Kitti/ImageSets/val.txt'
    val_ids = [x.strip() for x in open(val_dir).readlines()]
    ron_evaluation(val_ids, diff_list, cls_list, result_root)

    #write as file as well
    os.makedirs(f'KITTI_eval/{result_root.split("/")[0]}', exist_ok=True)
    org_stdout = sys.stdout
    f = open(f'KITTI_eval/{result_root}.txt', 'w')
    sys.stdout = f
    ron_evaluation(val_ids, diff_list, cls_list, result_root)
    sys.stdout = org_stdout
    f.close()
    print(f'save in KITTI_eval/{result_root}.txt')
    

if __name__ == '__main__':
    start = time.time()
    main()
    print('Done, take {} min'.format((time.time()-start)//60))# around 5min