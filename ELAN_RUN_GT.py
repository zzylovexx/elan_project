from torchvision.models import vgg
from torch_lib.Model_heading_bin import *
from torch_lib.ClassAverages import *
from torchvision import transforms
import os, glob, cv2
from library.ron_utils import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0, help='select cuda index')
# path setting
parser.add_argument("--weights-path", "-W_PATH", required=True, help='weights path, ie. weights/epoch_20.pkl')
parser.add_argument("--result-path", "-R_PATH", required=True, help='path (folder name) of the generated pred-labels')

def plot_regressed_3d_bbox(img, cam_to_img, box2d, dimensions, alpha, theta_ray, detectionid):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box2d, alpha, theta_ray)

    orient = alpha + theta_ray

    #plot_2d_box(img, box2d, detectionid)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location, orient

def main():

    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_path
    result_root = FLAGS.result_path
    os.makedirs(result_root, exist_ok=True)
    #os.makedirs(result_root+'/image_2', exist_ok=True)
    os.makedirs(result_root+'/label_2', exist_ok=True)

    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    checkpoint = torch.load(weights_path, map_location=device) #if training on 2 GPU, mapping on the same device
    normalize_type = 0#checkpoint['normal']
    bin_num = checkpoint['bin']
    angle_per_class = 2*np.pi/float(bin_num)

    my_vgg = vgg.vgg19_bn(weights='DEFAULT').to(device)
    model = Model(features=my_vgg.features, bins=bin_num).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # for img processing
    if normalize_type == 0:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if normalize_type == 1:
        normalize = transforms.Normalize(mean=[0.596, 0.612, 0.587], std=[0.256, 0.254, 0.257])
    process = transforms.Compose([transforms.ToTensor(), 
                              transforms.Resize([224,224], transforms.InterpolationMode.BICUBIC), 
                              normalize])

    # Kitti image_2 dir / label_2 dir
    img_root = "Elan_3d_box/image_2"
    label_root = "Elan_3d_box/renew_label"
    images = sorted(glob.glob(os.path.join(img_root, '*.png'), recursive=True))
    renew_labels = sorted(glob.glob(os.path.join(label_root, '*.txt'), recursive=True))
    # dim averages
    ELAN_averages = ClassAverages(average_file='all_ELAN_class_averages.txt')
    cam_to_img = np.array([
            [1.418667e+03, 0.000e+00, 6.4e+02, 0],
            [0.000e+00, 1.418867e+03, 3.6e+02, 0],
            [0.000e+00, 000e+00, 1.0e+00, 0] ])

    start = time.time()
    for i in range(len(renew_labels)):
        img = cv2.imread(images[i])
        lines = [x.strip() for x in open(renew_labels[i]).readlines()]
        label_ELAN = ''
        for idx, line in enumerate(lines):
            #print(line)
            elements = line.split()
            for j in range(1, len(elements)):
                elements[j] = float(elements[j])
            
            class_ = elements[0]
            truncate = elements[1]
            occluded = elements[2]
                
            top_left = (int(round(elements[4])), int(round(elements[5])))
            btm_right = (int(round(elements[6])), int(round(elements[7])))
            box2d = (top_left, btm_right)
            dim_gt = [elements[8], elements[9], elements[10]] # height, width, length
            crop = img[top_left[1]:btm_right[1]+1, top_left[0]:btm_right[0]+1] 
            crop = process(crop)
            #2dbox
            crop = torch.stack([crop]).to(device)
            #Location = [elements[11], elements[12], elements[13]]
            
            alpha_gt = elements[3]
            ry_gt = elements[14]
            theta_ray = ry_gt - alpha_gt
            
            [RESIDUALs, BIN_CONFs, delta_DIMs] = model(crop)
            bin_argmax = torch.max(BIN_CONFs, dim=1)[1]
            orient_residual = RESIDUALs[torch.arange(len(RESIDUALs)), bin_argmax] 
            Alphas = angle_per_class*bin_argmax + orient_residual #mapping bin_class and residual to get alpha
            alpha_Elan = float(Alphas[0].data)
            alpha_Elan = angle_correction(alpha_Elan)
            dim_Elan = delta_DIMs.cpu().data.numpy()[0, :]
            dim_Elan += ELAN_averages.get_item(class_)

            loc, ry = plot_regressed_3d_bbox(img, cam_to_img, box2d, dim_Elan, alpha_Elan, theta_ray, idx)

            label_ELAN += '{CLASS} {T:.1f} {O} {A:.2f} {left} {top} {right} {btm} {H:.2f} {W:.2f} {L:.2f} {X:.2f} {Y:.2f} {Z:.2f} {Ry:.2f}\n'.format(
                            CLASS=class_, T=truncate, O=occluded, A=alpha_Elan, left=box2d[0][0], top=box2d[0][1], right=box2d[1][0], btm=box2d[1][1],
                            H=dim_Elan[0], W=dim_Elan[1], L=dim_Elan[2], X=loc[0], Y=loc[1], Z=loc[2], Ry=ry)
        
        with open(renew_labels[i].replace(label_root, result_root + '/label_2'), 'w') as ELAN_f:
            ELAN_f.writelines(label_ELAN)

        #cv2.imwrite(images[i].replace(img_root, result_root + '/image_2'), img)
        
        if i%500==0:
            print(i)
    print('Done, take {} min {:.2f} sec'.format((time.time()-start)//60, (time.time()-start)%60))# around 2min

if __name__=='__main__':
    main()