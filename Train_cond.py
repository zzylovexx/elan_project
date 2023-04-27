from torch_lib.Dataset_4dim import *
from torch_lib.Model import Model, OrientationLoss,residual_loss

import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg,resnet18
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os

def main():
    # writer_name='test'
    writer=SummaryWriter()
    writer = SummaryWriter('./cond_log')
    # hyper parameters
    epochs = 40    #100

    batch_size = 16   #b8 is better than b64

    alpha = 0.5
    w = 0.4

    print("Loading all detected objects in dataset...")

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    # # reset the first layer 0407
    my_vgg.features[0] = nn.Conv2d(4, 64, (3,3), (1,1), (1,1))
    
    model = Model().cuda()
    opt_adam = torch.optim.Adam(model.parameters(), lr=0.000125)
    conf_loss_func = nn.CrossEntropyLoss().cuda()    
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    # load any previous weights

    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights_cond_test/'


    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass
    use_latest=False
    if use_latest==True:
        if latest_model is not None:
            checkpoint = torch.load(model_path + latest_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt_adam.load_state_dict(checkpoint['optimizer_state_dict'])
            first_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print('Found previous checkpoint: %s at epoch %s'%(latest_model, first_epoch))
            print('Resuming training....')



    total_num_batches = int(len(dataset) / batch_size)#len(dataset)=40570
    # progress_bar=tqdm.tqdm(range(0,epochs),dynamic_ncols=True,leave=True,desc='epoch')
    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        # batch_progress_bar=tqdm.tqdm(len(generator),dynamic_ncols=True,leave=True,desc='batch')
        for batch_pass,(local_batch, local_labels) in enumerate(generator):
            opt_adam.zero_grad()
            truth_orient_resdiual = local_labels['heading_resdiual'].float().cuda()
    
            truth_bin = local_labels['heading_class'].long().cuda()#這個角度在哪個class上
            truth_dim = local_labels['Dimensions'].float().cuda()

            local_batch=local_batch.float().cuda()

            [orient_residual, bin_conf, dim] = model(local_batch)

            orient_bin_loss=F.cross_entropy(bin_conf,truth_bin,reduction='mean')
            orient_redisual_loss,rediual_val=residual_loss(orient_residual,truth_bin,truth_orient_resdiual)
            #orient_loss = orient_loss_func(orient, truth_orient_resdiual, truth_label_class_bin)
            dim_loss = dim_loss_func(dim, truth_dim)

            # truth_conf = torch.max(truth_conf, dim=1)[1]
            # conf_loss = conf_loss_func(conf, truth_conf)

            loss_theta = orient_bin_loss + orient_redisual_loss
            loss =dim_loss*0.5 + loss_theta
            loss.backward()
            opt_adam.step()
            # batch_progress_bar.update()


            if passes % 10 == 0:
                writer.add_scalar('value/resdiual_val',rediual_val[0],epoch*len(generator)+batch_pass)
                writer.add_scalar('per_batch/dim_loss',dim_loss.item(),epoch*len(generator)+batch_pass)
                writer.add_scalar('per_batch/heading_cls_loss',orient_bin_loss.item(),epoch*len(generator)+batch_pass)
                writer.add_scalar('per_batch/heading_residual_loss',orient_redisual_loss.item(),epoch*len(generator)+batch_pass)
                print("--- epoch %s | batch %s/%s --- [loss: %s],[redisual_loss:%5s],[dim_loss:%5s],[bin_loss:%5s]" %(epoch, curr_batch, total_num_batches, loss.item(),orient_redisual_loss.item(),dim_loss.item(),orient_bin_loss.item()))
                passes = 0
            passes += 1
            curr_batch += 1
        writer.add_scalar('loss/dim_loss',dim_loss.item(),epoch)
        writer.add_scalar('loss/heading_cls_loss',orient_bin_loss.item(),epoch)
        writer.add_scalar('loss/heading_residual_loss',orient_redisual_loss.item(),epoch)
        # batch_progress_bar.close()
        # progress_bar.update()
        # save after every 10 epochs
        if epoch % 20 == 0:
            name = model_path + 'epoch_%s.pkl' % epoch
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_adam.state_dict(),
                    'loss': loss
                    }, name)
            print("====================")
    # writer.close()

if __name__=='__main__':
    main()
