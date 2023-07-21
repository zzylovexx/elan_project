from torchvision.models import vgg
from torch_lib.Model_heading_bin import *
from torch_lib.ClassAverages import *
from torchvision import transforms
import torch.nn.functional as F

from library.ron_utils import *
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights-path', required=True, help='weights path to save pkl, ie. weights/0720.pkl')
parser.add_argument('--device', type=int, default=0, help='choose cuda index')
parser.add_argument("--normal", type=int, default=0, help='-1:None 0:ImageNet, 1:ELAN')

def get_object_label(objects, bin_num=4):
    ELAN_averages = ClassAverages(average_file='renew_ELAN_class_averages.txt')
    label = dict()
    Heading_class = list()
    Residual = list()
    Dim_delta = list()
    TrackID = list()
    for obj in objects:
        heading_class, residual = angle2class(obj.alphas[0], bin_num)
        dim_delta = np.array(obj.dims[0]) - ELAN_averages.get_item(obj.class_)
        obj_id = obj.id
        Heading_class.append(heading_class)
        Residual.append(residual)
        Dim_delta.append(dim_delta)
        TrackID.append(obj_id)
    label['bin'] = torch.tensor(Heading_class)
    label['residual'] = torch.tensor(Residual)
    label['dim_delta'] = torch.tensor(np.array(Dim_delta))
    label['track_id'] = TrackID
    return label

def id_compare(now_id, last_id):
    now_id_list = list()
    last_id_list = list()
    for idx, id_ in enumerate(now_id):
        try:
            find_idx = last_id.index(id_)
        except:
            find_idx = -1

        if find_idx != -1:
            now_id_list.append(idx)
            last_id_list.append(find_idx)
    return now_id_list, last_id_list

def main():
    keep_same_seeds(2023)
    FLAGS = parser.parse_args()
    device = torch.device(f'cuda:{FLAGS.device}')
    normalize_type = FLAGS.normal
    trainset = [x.strip() for x in open('Elan_3d_box/ImageSets/train.txt').readlines()]
    bin_num = 4
    angle_per_class = 2*np.pi/float(bin_num)
    batch_size = 16

    my_vgg = vgg.vgg19_bn(weights='DEFAULT')
    model = Model(features=my_vgg.features, bins=bin_num).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    if normalize_type==0: # IMAGENET
        normal = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    elif normalize_type==1: # ELAN
        normal = transforms.Normalize(mean=[0.596, 0.612, 0.587], std=[0.256, 0.254, 0.257])
    process = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224]), normal])

    model.train()
    start = time.time()
    for epoch in range(50):
        batch_count = 0
        count = 0
        passes = 0
        for id_ in trainset:
            img = cv2.cvtColor(cv2.imread(f'Elan_3d_box/image_2/{id_}.png'), cv2.COLOR_BGR2RGB)
            label = f'Elan_3d_box/renew_label_obj/{id_}.txt'
            lines = [x.strip() for x in open(label).readlines()]
            obj_count = len(lines)
            batch_count += len(lines)

            if obj_count == 0:
                continue
            objects = [TrackingObject(line) for line in lines]
            crops = [process(img[obj.box2d[0][1]:obj.box2d[1][1]+1 ,obj.box2d[0][0]:obj.box2d[1][0]+1]) for obj in objects]
            crops = torch.stack(crops).to(device)

            gt_labels = get_object_label(objects, bin_num)
            gt_bin = gt_labels['bin'].to(device)
            gt_residual = gt_labels['residual'].to(device)
            gt_dim_delta = gt_labels['dim_delta'].to(device)
            now_id = gt_labels['track_id']

            [reg_residual, reg_bin, reg_dim_delta] = model(crops)

            bin_loss = F.cross_entropy(reg_bin, gt_bin, reduction='mean')
            residual_loss, _ = compute_residual_loss(reg_residual, gt_bin, gt_residual, device)
            dim_loss = F.l1_loss(reg_dim_delta, gt_dim_delta, reduction='mean')

            if count == 0:
                consist_loss = torch.tensor(0.0)
                count +=1
            else:
                now_id_list, last_id_list = id_compare(now_id, last_id)
                #print(now_id, last_id)
                if len(now_id_list) != 0:
                    #0719 added
                    now_dim_delta = reg_dim_delta[now_id_list]
                    last_dim_delta = last_dim_delta[last_id_list]
                    consist_loss = F.l1_loss(now_dim_delta, last_dim_delta, reduction='mean')*len(now_id_list)/len(now_id)
                else:
                    consist_loss = torch.tensor(0.0)
                #print(consist_loss)

            angle_loss = bin_loss + residual_loss
            loss = 0.6 * dim_loss + angle_loss + consist_loss

            last_bin = torch.clone(reg_bin).detach()
            last_residual = torch.clone(reg_residual).detach()
            last_dim_delta = torch.clone(reg_dim_delta).detach()
            last_id = now_id
            #https://zhuanlan.zhihu.com/p/65002487
            loss /= obj_count
            loss.backward()  # 计算梯度
            if(batch_count//batch_size)== 1:
                #print(batch_count)
                batch_count = 0 
                optimizer.step()        
                optimizer.zero_grad()
                passes += 1
            
            if passes % 100 == 0:
                print("--- epoch %s | passes %s --- [loss: %.3f],[bin_loss:%.3f],[residual_loss:%.3f],[dim_loss:%.3f],[consist_loss:%.3f]]" \
                        %(epoch+1, passes, loss.item(), bin_loss.item(), residual_loss.item(), dim_loss.item(), consist_loss.item()))
                
        if (epoch+1) % 10 == 0:
                name = FLAGS.weights_path.split('.')[0] + f'_{epoch+1}.pkl'
                print("====================")
                print ("Done with epoch %s!" % (epoch+1))
                print ("Saving weights as %s ..." % name)
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'bin': bin_num,
                        }, name)
                print("====================")
    print(f'Elapsed time: {(time.time()-start)//60} min')

if __name__ == '__main__':
    main()