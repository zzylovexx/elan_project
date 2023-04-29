'''
可以跟train_cond整理成一個
'''
from torch_lib.Dataset_heading_bin import *
from torch_lib.Model_heading_bin import Model, residual_loss
from library.ron_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=2023, help='keep seeds to represent same result')
# path setting
parser.add_argument("--weights-path", required=True, default='weights/bin4_dim4_group.pkl', help='weights_folder/weights_name.pkl')
parser.add_argument("--latest-weights", default=None, help='only input the weights-name.pkl') #in the same folder as above
parser.add_argument("--log-dir", default='log', help='tensorboard log-saved path')

#training setting
parser.add_argument("--device", type=int, default=0, help='select cuda index')
parser.add_argument("--epoch", type=int, default=20, help='epoch num')
parser.add_argument("--warm-up", type=int, default=10, help='warm up before adding group loss')

#parser.add_argument("--batch-size", type=int, default=16, help='batch size')

# hyper-parameter (group | bin | cond)
parser.add_argument("--bin", type=int, default=4, help='heading bin num')
parser.add_argument("--group", action='store_true', help='if True, add stdGroupLoss')
parser.add_argument("--cond", action='store_true', help='if True, 4-dim with theta_ray')
# TO BE ADDED (Loss的weights比例alpha, w of groupLoss )

# to keep the same random result
def keep_same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    
    FLAGS = parser.parse_args()
    keep_same_seeds(FLAGS.seed)
    group_training = FLAGS.group
    warm_up = FLAGS.warm_up #大約15個epoch收斂 再加入grouploss訓練
    device = torch.device(f'cuda:{FLAGS.device}') # 選gpu的index

    os.makedirs(FLAGS.log_dir, exist_ok=True)
    writer = SummaryWriter(FLAGS.log_dir)
    # hyper parameters
    epochs = FLAGS.epoch
    batch_size = 16 #64 worse than 8
    alpha = 0.6

    print("Loading all detected objects in dataset...")
        
    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path, condition=FLAGS.cond, num_heading_bin=FLAGS.bin)
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    if FLAGS.cond:
        print("< 4-dim input, Theta_ray as Condition >")
        my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
    model = Model(features=my_vgg.features, bins=FLAGS.bin).to(device)
    angle_per_class=2*np.pi/float(FLAGS.bin)

    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # milestones:調整lr的epoch數，gamma:decay factor (https://hackmd.io/@Hong-Jia/H1hmbNr1d)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_SGD, milestones=[i for i in range(10, epochs, 20)], gamma=0.5)
    dim_loss_func = nn.MSELoss().to(device)
    if group_training==True:
        print("< Train with GroupLoss >")
        group_loss_func = stdGroupLoss_mono

    first_epoch = 0
    passes = 0
    best = [0, 0] # epoch, best_mean
    weights_folder = FLAGS.weights_path.split('/')[0]
    os.makedirs(weights_folder, exist_ok=True)
    if FLAGS.latest_weights is not None:
        weights_path = os.path.join(weights_folder, FLAGS.latest_weights)
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        passes = checkpoint['passes']
        best = checkpoint['best']
        print('Found previous checkpoint: %s at epoch %s'%(weights_path, first_epoch))
        print('Resuming training....')

    total_num_batches = int(len(dataset) / batch_size)#len(dataset)=40570
    
    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        GT_alpha_list = list()
        pred_alpha_list = list()
        
        for local_batch, local_labels in generator:

            truth_orient_resdiual = local_labels['heading_resdiual'].float().to(device)
            truth_bin = local_labels['heading_class'].long().to(device)#這個角度在哪個class上
            truth_dim = local_labels['Dimensions'].float().to(device)

            local_batch=local_batch.float().to(device)
            [orient_residual, bin_conf, dim] = model(local_batch)

            bin_loss = F.cross_entropy(bin_conf,truth_bin,reduction='mean').to(device)
            orient_redisual_loss, rediual_val = residual_loss(orient_residual,truth_bin,truth_orient_resdiual, device)

            loss_theta = bin_loss + orient_redisual_loss
            dim_loss = dim_loss_func(dim, truth_dim)
            loss = alpha * dim_loss + loss_theta

            # return list type
            pred_alpha = angle_per_class*truth_bin + orient_residual[torch.arange(len(orient_residual)), truth_bin]
            GT_alpha = angle_per_class*truth_bin + truth_orient_resdiual
            pred_alpha_list += pred_alpha.tolist()
            GT_alpha_list += GT_alpha.tolist()
            
            #added loss
            if group_training==True and epoch > warm_up:
                truth_Theta = local_labels['Theta'].float().to(device)
                truth_Ry = local_labels['Ry'].float().to(device)
                truth_group = local_labels['Group'].float().to(device)
                group_loss = group_loss_func(pred_alpha, truth_Theta, truth_group, device)
                loss += 0.3 * group_loss
                
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            if passes % 200 == 0 and group_training==True and epoch> warm_up:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[bin_loss:%.4f],[redisual_loss:%.4f],[dim_loss:%.4f],[group_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(), bin_loss.item(), orient_redisual_loss.item(), dim_loss.item(), group_loss.item()))
                writer.add_scalar('pass/orient_cls_loss', bin_loss, passes//200)
                writer.add_scalar('pass/residual_loss', orient_redisual_loss, passes//200)
                writer.add_scalar('pass/dim_loss', dim_loss, passes//200)
                writer.add_scalar('pass/loss_theta', loss_theta, passes//200)
                writer.add_scalar('pass/total_loss', loss, passes//200) 
                writer.add_scalar('pass/group_loss', 0, passes//200) 

            elif passes % 200 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %.4f],[bin_loss:%.4f],[redisual_loss:%.4f],[dim_loss:%.4f]" \
                    %(epoch, curr_batch, total_num_batches, loss.item(), bin_loss.item(), orient_redisual_loss.item(),dim_loss.item()))
                writer.add_scalar('pass/orient_cls_loss', bin_loss, passes//200)
                writer.add_scalar('pass/residual_loss', orient_redisual_loss, passes//200)
                writer.add_scalar('pass/dim_loss', dim_loss, passes//200)
                writer.add_scalar('pass/loss_theta', loss_theta, passes//200)
                writer.add_scalar('pass/total_loss', loss, passes//200) 
                writer.add_scalar('pass/group_loss', 0, passes//200) 

            passes += 1
            curr_batch += 1

        cos_delta = angle_criterion(pred_alpha_list, GT_alpha_list)
        print(f'Epoch:{epoch} lr = {scheduler.get_last_lr()[0]}')
        print(f'cos_delta: sum={cos_delta.sum()}, mean:{cos_delta.mean()}') #sum=40570 is best, mean=1 is best
        # record the best_epoch and best_mean
        if cos_delta.mean() > best[1]:
            best = [epoch, cos_delta.mean()]
        writer.add_scalar('epoch/cos_delta_sum', cos_delta.sum(), epoch)
        writer.add_scalar('epoch/cos_delta_mean', cos_delta.mean(), epoch)
        #write every epoch
        writer.add_scalar('pass/orient_cls_loss', bin_loss, epoch)
        writer.add_scalar('pass/residual_loss', orient_redisual_loss, epoch)
        writer.add_scalar('pass/dim_loss', dim_loss, epoch)
        writer.add_scalar('epoch/loss_theta', loss_theta, epoch)
        writer.add_scalar('epoch/total_loss', loss, epoch) 
        if group_training==True and epoch > warm_up:
            writer.add_scalar('epoch/group_loss', group_loss, epoch)
        
        # visiualize https://zhuanlan.zhihu.com/p/103630393
        #tensorboard --logdir=./{log_foler} --port 8123

        # save after every 10 epochs
        scheduler.step()
        if epoch % 1 == 0:
            name = FLAGS.weights_path.split('.')[0] + f'_epoch_{epoch}.pkl'
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print(f'Best mean:{best[1]} @ epoch {best[0]}')
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss,
                    'passes': passes, # for continue training
                    'best': best,
                    'bin': FLAGS.bin, # for evaluate
                    'cond': FLAGS.cond # for evaluate
                    }, name)
            print("====================")
            
    writer.close()

if __name__=='__main__':
    main()
