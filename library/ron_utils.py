import torch
import numpy as np
from .Plotting import *

def calc_theta_ray(width, box_2d, proj_matrix):#透過跟2d bounding box 中心算出射線角度
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = (box_2d[1][0] + box_2d[0][0]) / 2
    dx = center - (width / 2)

    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
    angle = angle * mult

    return angle

def sign(num):
    return 1 if num>=0 else -1

def get_box_center(d2_box):
    return [(d2_box[0][0]+d2_box[1][0])//2, (d2_box[0][1]+d2_box[1][1])//2]

def get_box_size(d2_box):
    width = max(d2_box[1][0]-d2_box[0][0], 1)
    height = max(d2_box[1][1]-d2_box[0][1], 1)
    return width, height

def calc_offset(d2_box, d3_location, cam_to_img):
    d2_center = get_box_center(d2_box)
    proj_center = project_3d_pt(d3_location, cam_to_img)
    d2_box_size = get_box_size(d2_box)
    offset_x = proj_center[0] - d2_center[0] #pixel
    offset_y = proj_center[1] - d2_center[1]
    return offset_x, offset_y

#offset ratio -1~1
def calc_center_offset_ratio(d2_box, d3_location, cam_to_img):
    d2_center = get_box_center(d2_box)
    proj_center = project_3d_pt(d3_location, cam_to_img)
    d2_box_size = get_box_size(d2_box)
    
    if d2_box_size[0] < 10:
        offset_x = 0
    else:
        offset_x = (proj_center[0] - d2_center[0]) / (d2_box_size[0]//2)

    if d2_box_size[1] < 10:
        offset_y = 0
    else:
        offset_y = (proj_center[1] - d2_center[1]) / (d2_box_size[1]//2)
    
    if abs(offset_x) > 1: 
        offset_x = sign(offset_x)*1.0
    if abs(offset_y) > 1:
        offset_y = sign(offset_y)*1.0

    return [offset_x, offset_y]

def offset_to_projected_center(d2_box, offset_ratio):
    d2_center = get_box_center(d2_box)
    box_size = get_box_size(d2_box)
    proj_center = [0, 0]
    proj_center[0] = d2_center[0] + (offset_ratio[0]*box_size[0])//2
    proj_center[1] = d2_center[1] + (offset_ratio[1]*box_size[1])//2
    return np.array(proj_center, dtype=int)

#added 0417 RyGT_batch unused in std
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

#added 0417 -1*cos(GT_Ry-pred_Ry) , similar to OrientationLoss -1*cos(GT_alpha-pred_alpha)
def RyGroupLoss(orient_batch, confGT_batch, group_batch, ThetaGT_batch, RyGT_batch): #

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
            
        loss += -1 * torch.cos(RyGT_batch[idxs] - estimated_Ry[idxs]).sum()/batch_size 

    return loss.requires_grad_(True)

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

def get_alpha(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]#conf 是在那一個bin上取大
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    GT_alpha = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    PRED_alpha = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return GT_alpha.cpu().data.tolist(), PRED_alpha.cpu().data.tolist() # np array

def angle_criterion(PRED, GT):
    PRED = np.array(PRED)
    GT = np.array(GT)
    cos_delta = np.cos(PRED - GT)
    return cos_delta

def get_extra_labels(file_name):
    '''
    [Return] List[Dict]

    [Dict Keys] Group_Ry, Theta_ray, Img_W, Box_W, Box_H, Offset_X, Offset_Y, Group_Alpha 
    '''
    extra_label_list = list()
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            extra_label = dict()
            elements = line[:-1].split()
            extra_label['Img_W'] = int(elements[0])
            extra_label['Box_W'] = int(elements[1])
            extra_label['Box_H'] = int(elements[2])
            extra_label['Offset_X'] = int(elements[3])
            extra_label['Offset_Y'] = int(elements[4])
            extra_label['Theta_ray'] = float(elements[5]) # (.3f)
            extra_label['Group_Alpha'] = int(elements[6]) # (+100*index)
            extra_label['Group_Ry'] = int(elements[7])    # (+100*index)
            extra_label_list.append(extra_label)
    return extra_label_list
            
