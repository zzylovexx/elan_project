from torchvision.models import vgg
from torch_lib.Model_heading_bin import *
from torch_lib.ClassAverages import *
from torchvision import transforms
import os, glob, cv2, json
from library.ron_utils import *
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0, help='select cuda index')
# path setting
parser.add_argument("--weights-path", "-W_PATH", required=True, help='weights path, ie. weights/epoch_20.pkl')
parser.add_argument('--data-path', "-D_PATH", required=True, help='folder of the elan dataset')

def main():

    FLAGS = parser.parse_args()
    weights_path = FLAGS.weights_path
    data_root = FLAGS.data_path

    weights_name = weights_path.split('weights/')[1].split('.')[0]
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    checkpoint = torch.load(weights_path, map_location=device) #if training on 2 GPU, mapping on the same device
    normalize_type = 0#checkpoint['normal']
    bin_num = checkpoint['bin']
    angle_per_class = 2*np.pi/float(bin_num)

    my_vgg = vgg.vgg19_bn(weights='DEFAULT').to(device)
    model = vgg_Model(features=my_vgg.features, bins=bin_num).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # for img processing
    if normalize_type == 0:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if normalize_type == 1:
        normalize = transforms.Normalize(mean=[0.596, 0.612, 0.587], std=[0.256, 0.254, 0.257])
    process = transforms.Compose([transforms.ToTensor(), 
                              transforms.Resize([224,224], transforms.InterpolationMode.BICUBIC), 
                              normalize])

    # Kitti image_2 dir / label_2 dir
    print(data_root)
    # dim averages
    ELAN_averages = ClassAverages(average_file='all_ELAN_class_averages.txt')
    # 5:行人,6:機車騎士,7:腳踏車騎士,8:大車,9:小車,10:機車,11:腳踏車
    class_dict = {8:'truck', 9:'car', 10:'motor'} #目前只有這3個class
    cam_to_img = np.array([
            [1.418667e+03, 0.000e+00, 6.4e+02, 0],
            [0.000e+00, 1.418867e+03, 3.6e+02, 0],
            [0.000e+00, 000e+00, 1.0e+00, 0] ])

    model.eval()
    all_class_dict = dict()
    all_class_dict[8]=0
    all_class_dict[9]=0
    all_class_dict[10]=0
    with torch.no_grad():
        for sub_f in sorted(os.listdir(data_root)):
            sub_folder = os.path.join(data_root, sub_f)
            print(sub_folder)
            os.makedirs(os.path.join(sub_folder, weights_name), exist_ok=True)
            img_save_folder = os.path.join(sub_folder, weights_name, 'img')
            os.makedirs(img_save_folder, exist_ok=True)

            with open(f'{sub_folder}/TEST_images.json') as f:
                img_paths = json.load(f)
            label_json = f'{sub_folder}/TEST_objects.json'
            with open(label_json) as f:
                label_dicts = json.load(f)
            
            result_folder = list()
            result_json = os.path.join(sub_folder, weights_name, 'REG_result.json')

            all_class_dict = dict()
            all_class_dict[8]=0
            all_class_dict[9]=0
            all_class_dict[10]=0
            for i in range(len(img_paths)):
                path = os.path.join(data_root, img_paths[i])
                #print(path)
                img = cv2.imread(path)
                label_dict = label_dicts[i]
                img_save_path = path.replace('img', weights_name+ '/img')

                result_dict = dict()
                result_dict['alpha'] = list()
                result_dict['theta_ray'] = list()
                result_dict['dimension'] = list()
                result_dict['location'] = list()
                for j in range(len(label_dict['labels'])):
                    difficulty = label_dict['difficulties'][j] # 1 is hard to distinguish                    
                    if label_dict['labels'][j] not in class_dict.keys(): #8,9,10,11
                        continue
                    class_ = class_dict[label_dict['labels'][j]] 
                    all_class_dict[label_dict['labels'][j]] +=1
                    #object_id = label_dict['id'][j] # for tracking
                    box2d = label_dict['boxes'][j]
                    box2d = [[box2d[0], box2d[1]], [box2d[2],box2d[3]]]
                    #box_center = get_box_center(box2d)
                    left_top = box2d[0]
                    right_btm = box2d[1]

                    crop = img[left_top[1]:right_btm[1]+1, left_top[0]:right_btm[0]+1] 
                    crop = process(crop) 
                    crop = torch.stack([crop]).to(device)

                    [RESIDUALs, BIN_CONFs, delta_DIMs] = model(crop)
                    bin_argmax = torch.max(BIN_CONFs, dim=1)[1]
                    orient_residual = RESIDUALs[torch.arange(len(RESIDUALs)), bin_argmax] 
                    Alphas = angle_per_class*bin_argmax + orient_residual #mapping bin_class and residual to get alpha
                    reg_alpha = float(Alphas[0].data)
                    reg_alpha = round(angle_correction(reg_alpha), 2)
                    reg_dim = delta_DIMs.cpu().data.numpy()[0, :]
                    reg_dim += ELAN_averages.get_item(class_)
                    
                    theta_ray = round(calc_theta_ray(img.shape[1], box2d, cam_to_img), 2)
                    loc, ry = plot_regressed_3d_bbox(img, cam_to_img, box2d, reg_dim, reg_alpha, theta_ray)
                    txt_pos = [left_top[0], right_btm[1]]
                    if txt_pos[0]<10:
                        txt_pos[0]=0
                    if txt_pos[1]>img.shape[1]-10:
                        txt_pos[1]=img.shape[1]-10
                    cv2.putText(img, f'{round(ry,2)}', (left_top[0], right_btm[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    reg_dim = round_list(reg_dim, 2)
                    loc = round_list(loc, 2)
                    result_dict['alpha'].append(reg_alpha)
                    result_dict['theta_ray'].append(theta_ray)
                    result_dict['dimension'].append(reg_dim)
                    result_dict['location'].append(loc)
                    cv2.imwrite(img_save_path, img)

                result_folder.append(result_dict)
                
            print(all_class_dict)
            with open(result_json, 'w') as f:
                json.dump(result_folder, f)
            
            if i%500==0:
                print(i)



def plot_regressed_3d_bbox(img, cam_to_img, box2d, dimensions, alpha, theta_ray, detectionid=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box2d, alpha, theta_ray)

    orient = alpha + theta_ray
    
    if detectionid!=None:
        plot_2d_box(img, box2d, detectionid)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location, orient     

def round_list(list_, float_point=2):
    foo = list()
    for i in range(len(list_)):
        new_val = round(float(list_[i]), float_point)
        foo.append(new_val)
    return foo

# python ELAN_json_RUN_GT.py -W_PATH=weights/0830_Adam/Elan_BL_B4_N1_50.pkl -D_PATH=IVA_scenario_data_NCTU
if __name__=='__main__':
    start = time.time()
    main()
    print('Done, take {} min'.format((time.time()-start)//60))# around 2min