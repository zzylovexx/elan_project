from torchvision.models import vgg
from torch_lib.Model_heading_bin import *
from torch_lib.ClassAverages import *
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from library.ron_utils import *
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=2023, help='keep seeds to represent same result')
#path setting
parser.add_argument("--weights-path", "-W_PATH", required=True, help='folder/date ie.weights/0721')

#training setting
parser.add_argument("--type", "-T", type=int, default=0, help='0:dim, 1:alpha, 2:both')
parser.add_argument("--device", "-D", type=int, default=0, help='select cuda index')
parser.add_argument("--epoch", "-E", type=int, default=50, help='epoch num')

#parser.add_argument("--batch-size", type=int, default=16, help='batch size')

# hyper-parameter (group | bin | cond)
parser.add_argument("--normal", "-N", type=int, default=0, help='0:ImageNet, 1:ELAN')
parser.add_argument("--bin", "-B", type=int, default=4, help='heading bin num')
parser.add_argument("--group", "-G", type=int, help='if True, add stdGroupLoss')
parser.add_argument("--warm-up", "-W", type=int, default=10, help='warm up before adding group loss')
parser.add_argument("--cond", "-C", type=int, help='if True, 4-dim with theta_ray | boxH_2d ')

def main():

    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    type = FLAGS.type
    epochs = FLAGS.epoch
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index
    normalize_type = FLAGS.normal

    save_path, log_path, train_config = name_by_parameters(FLAGS)
    print(f'SAVE PATH:{save_path}, LOG PATH:{log_path}, config:{train_config}')
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    # weights 
    W_consist = 0.3
    W_alpha = 0.1
    W_group = 0.3

    #trainset = [x.strip() for x in open('Elan_3d_box/ImageSets/train.txt').readlines()]
    trainset = [x.strip() for x in open('Elan_3d_box/ImageSets/trainval.txt').readlines()]
    valset = [x.strip() for x in open('Elan_3d_box/ImageSets/val.txt').readlines()]

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
        acc_total_loss = torch.tensor(0.0).to(device)
        acc_dim_loss = torch.tensor(0.0).to(device)
        acc_residual_loss = torch.tensor(0.0).to(device)
        acc_bin_loss = torch.tensor(0.0).to(device)
        acc_group_loss = torch.tensor(0.0).to(device)
        acc_consist_loss = torch.tensor(0.0).to(device)
        acc_angle_loss = torch.tensor(0.0).to(device)

        for id_ in trainset:
            img = cv2.cvtColor(cv2.imread(f'Elan_3d_box/image_2/{id_}.png'), cv2.COLOR_BGR2RGB)
            label = f'Elan_3d_box/renew_label_obj/{id_}.txt'
            extra_labels = get_extra_labels(f'Elan_3d_box/extra_label/{id_}.txt')
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

            bin_loss = F.cross_entropy(reg_bin, gt_bin, reduction='mean').to(device)
            residual_loss, _ = compute_residual_loss(reg_residual, gt_bin, gt_residual, device)
            dim_loss = F.l1_loss(reg_dim_delta, gt_dim_delta, reduction='mean').to(device)

            # GROUP LOSS
            reg_alphas = compute_alpha(reg_bin, reg_residual, angle_per_class)
            if is_group and epoch > warm_up:
                for i in range(len(objects)):
                    gt_labels['Group'].append(extra_labels[i]['Group_Ry'])
                    gt_labels['Theta'].append(extra_labels[i]['Theta_ray'])
                gt_Theta = torch.tensor(gt_labels['Theta']).to(device)
                #gt_Ry = gt_labels['Ry'].float().to(device)
                gt_group = torch.tensor(gt_labels['Group']).to(device)
                group_loss = stdGroupLoss_heading_bin(reg_alphas, gt_Theta, gt_group, device)
            else:
                group_loss = torch.tensor(0.0)
            
            loss = 0.6*dim_loss + bin_loss + residual_loss + W_group*group_loss # before0724 W_group=0.3
    
            # CONSISTENCY LOSS
            if count == 0:
                consist_loss = torch.tensor(0.0)
                angle_loss = torch.tensor(0.0)
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
                    now_alphas = reg_alphas[now_id_list]
                    last_alphas = compute_alpha(last_bin, last_residual, angle_per_class)[last_id_list]
                    angle_loss = F.l1_loss(torch.cos(now_alphas), torch.cos(last_alphas), reduction='mean')*len(now_id_list)/len(now_id)
                else:
                    consist_loss = torch.tensor(0.0)
                    angle_loss = torch.tensor(0.0)
            
            if type==0:
                angle_loss = torch.tensor(0.0)
            elif type==1:
                consist_loss = torch.tensor(0.0)

            loss += W_consist*consist_loss.to(device) + W_alpha*angle_loss.to(device) # W_consist=1, W_alpha=0.1 before 0723

            last_bin = torch.clone(reg_bin).detach()
            last_residual = torch.clone(reg_residual).detach()
            last_dim_delta = torch.clone(reg_dim_delta).detach()
            last_id = now_id
            #https://zhuanlan.zhihu.com/p/65002487
            loss /= obj_count
            loss.backward()  # 计算梯度

            acc_total_loss += loss * obj_count
            acc_bin_loss += bin_loss
            acc_residual_loss += residual_loss
            acc_dim_loss += dim_loss
            acc_group_loss += group_loss
            acc_consist_loss += consist_loss
            acc_angle_loss += angle_loss
            
            if(batch_count//batch_size)== 1:
                #print(batch_count)
                batch_count = 0 
                optimizer.step()        
                optimizer.zero_grad()
                passes += 1
            
            if passes % 100 == 0:
                print("--- epoch %s | passes %s --- [loss: %.3f],[bin_loss:%.3f],[residual_loss:%.3f],[dim_loss:%.3f],[consist_loss:%.3f],[angle_loss:%.3f]" \
                        %(epoch, passes, loss.item(), bin_loss.item(), residual_loss.item(), dim_loss.item(), consist_loss.item(), angle_loss.item()))
                if is_group and epoch > warm_up:
                    print("[group_loss:%.3f]"%(group_loss.item()))
        
        # record to tensorboard
        writer.add_scalar(f'{train_config}/total', acc_total_loss, epoch) 
        writer.add_scalar(f'{train_config}/bin', acc_bin_loss, epoch)
        writer.add_scalar(f'{train_config}/residual', acc_residual_loss, epoch)
        writer.add_scalar(f'{train_config}/dim', acc_dim_loss, epoch)
        writer.add_scalar(f'{train_config}/group', group_loss, epoch)
        writer.add_scalar(f'{train_config}/consist', acc_consist_loss, epoch)
        writer.add_scalar(f'{train_config}/angle', acc_angle_loss, epoch)
        '''
        if type!=1:
            writer.add_scalars(f'{train_config}/DIM', {'dim': acc_dim_loss, 'consist': acc_consist_loss}, epoch)
        if type!=0:
            writer.add_scalars(f'{train_config}/Alpha', {'bin': acc_bin_loss, 'residual': acc_residual_loss,'angle_loss': acc_angle_loss}, epoch)
        '''
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        # MobaXterm https://zhuanlan.zhihu.com/p/138811263
        #tensorboard --logdir=./{log_foler} --port 8123

        #
        model.eval()
        GT_alpha_list = list()
        REG_alpha_list = list()
        with torch.no_grad():
            for id_ in valset:
                img = cv2.cvtColor(cv2.imread(f'Elan_3d_box/image_2/{id_}.png'), cv2.COLOR_BGR2RGB)
                label = f'Elan_3d_box/renew_label_obj/{id_}.txt'
                extra_labels = get_extra_labels(f'Elan_3d_box/extra_label/{id_}.txt')
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

                [reg_residual, reg_bin, reg_dim_delta] = model(crops)
                bin_argmax = torch.max(reg_bin, dim=1)[1]
                reg_alpha = angle_per_class*bin_argmax + reg_residual[torch.arange(len(reg_residual)), bin_argmax]
                GT_alpha = angle_per_class*gt_bin + gt_residual
                REG_alpha_list += reg_alpha.tolist()
                GT_alpha_list += GT_alpha.tolist()
        
        alpha_performance = angle_criterion(REG_alpha_list, GT_alpha_list)
        print(f'alpha_performance: {alpha_performance:.4f}') #close to 0 is better\
        writer.add_scalar(f'{train_config}/alpha_eval', alpha_performance, epoch)
        #write every epoch
            
        if epoch % (epochs//2) == 0:
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
                        'normal': normalize_type,
                        'W_consist': W_consist,
                        'W_group': W_group,
                        }, name)
                print("====================")
    writer.close()
    print(f'Elapsed time: {(time.time()-start)//60} min')

def name_by_parameters(FLAGS):
    is_group = FLAGS.group
    is_cond = FLAGS.cond
    bin_num = FLAGS.bin
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    normalize_type = FLAGS.normal
    
    save_path = f'{FLAGS.weights_path}_B{bin_num}_N{normalize_type}'
    if is_group==1:
        save_path += f'_G_W{warm_up}'
    if is_cond==1:
        save_path += '_C'

    train_config = save_path.split("weights/")[1]
    log_path = f'log/{train_config}'

    return save_path, log_path, train_config

def get_object_label(objects, bin_num=4):
    ELAN_averages = ClassAverages(average_file='all_ELAN_class_averages.txt')
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
    label['Group'] = list()
    label['Theta'] = list()
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

if __name__ == '__main__':
    main()