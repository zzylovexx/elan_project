from torchvision.models import vgg
from torch_lib.Model_heading_bin import *
from torch_lib.ClassAverages import *
from torchvision import transforms
import torch.nn.functional as F

from library.ron_utils import *
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=2023, help='keep seeds to represent same result')
#path setting
parser.add_argument("--weights-path", "-W_PATH", required=True, help='folder/date ie.weights/0721')

#training setting
parser.add_argument("--device", "-D", type=int, default=0, help='select cuda index')
parser.add_argument("--epoch", "-E", type=int, default=50, help='epoch num')

#parser.add_argument("--batch-size", type=int, default=16, help='batch size')

# hyper-parameter (group | bin | cond)
parser.add_argument("--normal", "-N", type=int, default=0, help='0:ImageNet, 1:ELAN')
parser.add_argument("--bin", "-B", type=int, default=4, help='heading bin num')
parser.add_argument("--group", "-G", type=int, help='if True, add stdGroupLoss')
parser.add_argument("--warm-up", "-W", type=int, default=10, help='warm up before adding group loss')
parser.add_argument("--cond", "-C", type=int, help='if True, 4-dim with theta_ray | boxH_2d ')


def get_object_label(objects, bin_num=4):
    ELAN_averages = ClassAverages(average_file='ELAN_class_averages.txt')
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

def compute_alpha(bin, residual, angle_per_class):
    bin_argmax = torch.max(bin, dim=1)[1]
    residual = residual[torch.arange(len(residual)), bin_argmax] 
    alphas = angle_per_class*bin_argmax + residual #mapping bin_class and residual to get alpha
    for i in range(len(alphas)):
        alphas[i] = angle_correction(alphas[i])
    return alphas

def main():

    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    epochs = FLAGS.epoch
    is_group = FLAGS.group
    
    is_cond = FLAGS.cond
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    normalize_type = FLAGS.normal

    save_path = f'{FLAGS.weights_path}Vdim_alpha_B{bin_num}_N{normalize_type}'
    if is_group==1:
        save_path += f'_G_W{warm_up}'
    if is_cond==1:
        save_path += '_C'
    print(save_path)

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
    for epoch in range(1, epochs+1):
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
                consist_angle_loss = torch.tensor(0.0)
                count +=1
            else:
                now_id_list, last_id_list = id_compare(now_id, last_id)
                #print(now_id, last_id)
                if len(now_id_list) != 0:
                    #0719 added
                    now_dim_delta = reg_dim_delta[now_id_list]
                    last_dim_delta = last_dim_delta[last_id_list]
                    consist_loss = F.l1_loss(now_dim_delta, last_dim_delta, reduction='mean')*len(now_id_list)/len(now_id)
                    #0721 added
                    now_alphas = compute_alpha(reg_bin, reg_residual, angle_per_class)[now_id_list]
                    last_alphas = compute_alpha(last_bin, last_residual, angle_per_class)[last_id_list]
                    consist_angle_loss = F.l1_loss(torch.cos(now_alphas), torch.cos(last_alphas), reduction='mean')*len(now_id_list)/len(now_id)
                else:
                    consist_loss = torch.tensor(0.0)
                    consist_angle_loss = torch.tensor(0.0)
                #print(consist_loss)

            angle_loss = bin_loss + residual_loss
            loss = 0.6 * dim_loss + angle_loss + consist_loss.to(device) + 0.1*consist_angle_loss.to(device)

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
                print("--- epoch %s | passes %s --- [loss: %.3f],[bin_loss:%.3f],[residual_loss:%.3f],[dim_loss:%.3f],[consist_loss:%.3f],[consist_angle_loss:%.3f]" \
                        %(epoch, passes, loss.item(), bin_loss.item(), residual_loss.item(), dim_loss.item(), consist_loss.item(), consist_angle_loss.item()))
                
        if epoch % 50 == 0:
                name = save_path + f'_{epoch}.pkl'
                print("====================")
                print ("Done with epoch %s!" % (epoch))
                print ("Saving weights as %s ..." % name)
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'bin': bin_num,
                        'cond': is_cond,
                        'normal': normalize_type
                        }, name)
                print("====================")
    print(f'Elapsed time: {(time.time()-start)//60} min')

if __name__ == '__main__':
    main()