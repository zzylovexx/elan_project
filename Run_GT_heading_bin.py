from torchvision.models import vgg
from torch_lib.Model_heading_bin import *
from torch_lib.ClassAverages import *
from torchvision import transforms
import os, glob, cv2
from library.File import get_calibration_cam_to_image
from library.ron_utils import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0, help='select cuda index')
# path setting
parser.add_argument("--weights-path", required=True, default='weights/epoch_20.pkl', help='weighs path')
parser.add_argument("--result-path", required=True, default='Result', help='path (folder name) of the generated pred-labels')

def main():

    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_path
    result_root = FLAGS.result_path
    os.makedirs(result_root, exist_ok=True)

    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    checkpoint = torch.load(weights_path, map_location=device) #if training on 2 GPU, mapping on the same device
    bin_num = checkpoint['bin'] 
    is_cond = checkpoint['cond']
    angle_per_class = 2*np.pi/float(bin_num)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    if is_cond:
        print("< add Condition (4-dim) as input >")
        my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
    model = Model(features=my_vgg.features, bins=bin_num).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # for img processing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    process = transforms.Compose([transforms.ToTensor(), normalize])

    # Kitti image_2 dir / label_2 dir
    img_root = "./Kitti/training/image_2"
    label_root = "./Kitti/training/label_2"
    calib_root = "./Kitti/training/calib"
    extra_label_root = "./Kitti/training/extra_label"
    ImageSets_root = './Kitti/ImageSets'
    split = 'trainval'

    images = glob.glob(os.path.join(img_root, '*.png'), recursive=True)
    labels = glob.glob(os.path.join(label_root, '*.txt'), recursive=True)
    calibs = glob.glob(os.path.join(calib_root, '*.txt'), recursive=True)
    extra = glob.glob(os.path.join(extra_label_root, '*.txt'), recursive=True)

    split_dir = os.path.join(ImageSets_root, split + '.txt')
    ids = [int(x.strip()) for x in open(split_dir).readlines()]

    # dim averages
    averages_all = ClassAverages()
    start = time.time()
    for i in ids:
        img = cv2.imread(images[i])
        img_W = img.shape[1]
        cam_to_img = get_calibration_cam_to_image(calibs[i])

        CLASSes = list()
        TRUNCATEDs = list()
        OCCLUDEDs = list()
        BOX2Ds = list()
        CROPs_tensor = list()
        Alphas = list()
        THETAs = list()
        depth_GT = list()
        extra_labels = get_extra_labels(extra[i])

        with open(labels[i]) as f:
            lines = f.readlines()

            for idx, line in enumerate(lines):
                elements = line[:-1].split()
                if elements[0] == 'DontCare':
                    continue
                for j in range(1, len(elements)):
                    elements[j] = float(elements[j])

                CLASSes.append(elements[0])
                TRUNCATEDs.append(elements[1])
                OCCLUDEDs.append(elements[2])
                top_left = (int(round(elements[4])), int(round(elements[5])))
                btm_right = (int(round(elements[6])), int(round(elements[7])))
                box = [top_left, btm_right]
                BOX2Ds.append(box)
                #cv2 is(H,W,3)
                crop = img[top_left[1]:btm_right[1]+1, top_left[0]:btm_right[0]+1] 
                crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                crop = process(crop) # expand to 224x224

                # Use 'calc_theta_ray(img_width, box, proj_matrix)', if cam_to_img changes to proj_matrix:"camera_cal/calib_cam_to_cam.txt"
                theta_ray = extra_labels[idx]['Theta_ray']
                THETAs.append(theta_ray)
                depth_label = elements[13]
                depth_GT.append(depth_label)
                #4dim
                if is_cond:
                    cond = torch.tensor(theta_ray).expand(1, crop.shape[1], crop.shape[2])
                    img_cond = torch.concat((crop, cond), dim=0) # 3+1, 224, 224
                    CROPs_tensor.append(img_cond)
                else:
                    CROPs_tensor.append(crop)

            # put together as a batch
            input_ = torch.stack(CROPs_tensor).to(device)
            # model regress part
            [RESIDUALs, BIN_CONFs, delta_DIMs] = model(input_)

            bin_argmax = torch.max(BIN_CONFs, dim=1)[1]
            orient_residual = RESIDUALs[torch.arange(len(RESIDUALs)), bin_argmax] 
            Alphas = angle_per_class*bin_argmax + orient_residual #mapping bin_class and residual to get alpha

        #write pred_label.txt 
        with open(labels[i].replace(label_root, result_root),'w') as new_f:
            pred_labels = ''
            for class_, truncated, occluded, delta, alpha, theta, box_2d, depth_gt in zip(CLASSes, TRUNCATEDs, OCCLUDEDs, delta_DIMs, Alphas, THETAs, BOX2Ds, depth_GT):
                delta = delta.cpu().data #torch->numpy
                alpha = alpha.cpu().data #torch->numpy
                alpha = angle_correction(alpha)
                dim = delta + averages_all.get_item(class_)
                rotation_y = alpha + theta
                loc, _ = calc_location(dim, cam_to_img, box_2d, alpha, theta)    
                pred_labels += '{CLASS} {T:.1f} {O} {A:.2f} {left} {top} {right} {btm} {H:.2f} {W:.2f} {L:.2f} {X:.2f} {Y:.2f} {Z:.2f} {Ry:.2f}\n'.format(
                    CLASS=class_, T=truncated, O=occluded, A=alpha, left=box_2d[0][0], top=box_2d[0][1], right=box_2d[1][0], btm=box_2d[1][1],
                    H=dim[0], W=dim[1], L=dim[2], X=loc[0], Y=loc[1], Z=loc[2], Ry=rotation_y)

            #print(pred_labels)
            new_f.writelines(pred_labels)
        if i%500==0:
            print(i)
    #print('Done, take {} min {} sec'.format((time.time()-start)//60, (time.time()-start)%60))# around 10min

if __name__=='__main__':
    main()