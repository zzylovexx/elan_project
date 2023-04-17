import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]#conf 是在那一個bin上取大

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

#added 0417 RyGT_batch unused now
def stdGroupLoss(orient_batch, confGT_batch, group_batch, ThetaGT_batch, RyGT_batch): #

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]#conf 是在那一個bin上取大
    # extract just the important bin
    orient_batch = orient_batch[torch.arange(batch_size), indexes]
    estimated_alpha = torch.atan2(orient_batch[:,1], orient_batch[:,0])
    estimated_Ry = estimated_alpha + ThetaGT_batch  

    group_idxs = get_group_idxs(group_batch)
    loss = torch.zeros(1)[0].cuda()
    for idxs in group_idxs:
        if len(idxs) == 1:
            continue
            
        value_tensor_list = estimated_Ry[idxs]
        stddev = torch.std(value_tensor_list)
        loss += stddev*len(idxs)/batch_size

    return loss.requires_grad_(True)

def GroupLoss(orient_batch, orientGT_batch, confGT_batch, group_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]#conf 是在那一個bin上取大

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]
    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    group_idxs = get_group_idxs(group_batch)
    weighted_loss = 0
    for idxs in group_idxs:
        loss = eachGroupLoss_sin(idxs, theta_diff, estimated_theta_diff)
        #loss = eachGroupLoss_abssin(idxs, theta_diff, estimated_theta_diff)
        #loss = eachGroupLoss_cos(idxs, theta_diff, estimated_theta_diff)
        #loss = eachGroupLoss_cos_1(idxs, theta_diff, estimated_theta_diff) # try on 0303
        weighted_loss += loss*len(idxs)/batch_size

    #return torch.tensor(weighted_loss)
    return weighted_loss.requires_grad_(True)

#theta_loss = torch.cos(theta_diff - estimated_theta_diff)
def eachGroupLoss_sin(idxs, theta_diff, estimated_theta_diff):
    if len(idxs) == 1:
        return torch.zeros(1)[0].cuda()
    
    theta_diff = theta_diff[idxs]
    estimated_theta_diff = estimated_theta_diff[idxs]
    delta = torch.cos(theta_diff - estimated_theta_diff)
    closest_idx = torch.argmax(delta)
    group_theta_diff = estimated_theta_diff - estimated_theta_diff[closest_idx]
    group_theta_loss = torch.sin(group_theta_diff).sum()
    return group_theta_loss

def eachGroupLoss_abssin(idxs, theta_diff, estimated_theta_diff):
    if len(idxs) == 1:
        return torch.zeros(1)[0].cuda()
    
    theta_diff = theta_diff[idxs]
    estimated_theta_diff = estimated_theta_diff[idxs]
    delta = torch.cos(theta_diff - estimated_theta_diff)
    closest_idx = torch.argmax(delta)
    group_theta_diff = estimated_theta_diff - estimated_theta_diff[closest_idx]
    group_theta_diff = torch.abs(group_theta_diff)
    group_theta_loss = torch.sin(group_theta_diff).sum()
    return group_theta_loss

def eachGroupLoss_cos(idxs, theta_diff, estimated_theta_diff):
    if len(idxs) == 1:
        return torch.zeros(1)[0].cuda()
    
    theta_diff = theta_diff[idxs]
    estimated_theta_diff = estimated_theta_diff[idxs]
    theta_loss = torch.cos(theta_diff - estimated_theta_diff)
    closest_idx = torch.argmax(theta_loss)
    group_theta_diff = estimated_theta_diff - estimated_theta_diff[closest_idx]
    # sin0 = 0  
    return -1 * torch.cos(group_theta_diff).sum()

def eachGroupLoss_cos_1(idxs, theta_diff, estimated_theta_diff):
    if len(idxs) == 1:
        return torch.zeros(1)[0].cuda()
    
    theta_diff = theta_diff[idxs]
    estimated_theta_diff = estimated_theta_diff[idxs]
    theta_loss = torch.cos(theta_diff - estimated_theta_diff)
    closest_idx = torch.argmax(theta_loss)
    group_theta_diff = estimated_theta_diff - estimated_theta_diff[closest_idx]
    # cos0 = 1
    return -1 * (torch.cos(group_theta_diff).sum()-1)

def get_group_idxs(group):
    # remove duplicate values
    values = list(dict.fromkeys(group.tolist()))
    group_idxs = list()
    for val in values:
        tmp = list()
        for i in range(len(group)):
            if group[i] == val:
                tmp.append(i)
        group_idxs.append(tmp)
    return group_idxs

class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins #2
        self.w = w #0.4
        self.features = features
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
                    # nn.Softmax()
                    #nn.Sigmoid()
                )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )

    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation, confidence, dimension
