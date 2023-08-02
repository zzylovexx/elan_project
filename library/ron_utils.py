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
def stdGroupLoss(orient_batch, confGT_batch, group_batch, ThetaGT_batch, RyGT_batch, device): #

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]#conf 是在那一個bin上取大
    # extract just the important bin
    orient_batch = orient_batch[torch.arange(batch_size), indexes]
    estimated_alpha = torch.atan2(orient_batch[:,1], orient_batch[:,0])
    estimated_Ry = estimated_alpha + ThetaGT_batch  

    group_idxs = get_group_idxs(group_batch)
    loss = torch.zeros(1)[0].to(device)
    for idxs in group_idxs:
        if len(idxs) == 1:
            continue
            
        value_tensor_list = estimated_Ry[idxs]
        stddev = torch.std(value_tensor_list)
        loss += stddev*len(idxs)/batch_size

    return loss.requires_grad_(True)

#added 0417 RyGT_batch unused in std
#stdGroupLoss_heading_bin
def stdGroupLoss_heading_bin(pred_alpha, truth_Theta, group_batch, device): #

    batch_size = pred_alpha.shape[0]
    estimated_Ry = pred_alpha + truth_Theta  

    group_idxs = get_group_idxs(group_batch)
    loss = torch.tensor(0.0).to(device)
    for idxs in group_idxs:
        if len(idxs) == 1:
            continue
            
        value_tensor_list = estimated_Ry[idxs]
        stddev = torch.std(value_tensor_list)
        loss += stddev*len(idxs)/batch_size

    return loss.requires_grad_(True)

#added 0417 -1*cos(GT_Ry-pred_Ry) , similar to OrientationLoss -1*cos(GT_alpha-pred_alpha)
def RyGroupLoss(orient_batch, confGT_batch, group_batch, ThetaGT_batch, RyGT_batch, device): #

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]#conf 是在那一個bin上取大
    # extract just the important bin
    orient_batch = orient_batch[torch.arange(batch_size), indexes]
    estimated_alpha = torch.atan2(orient_batch[:,1], orient_batch[:,0])
    estimated_Ry = estimated_alpha + ThetaGT_batch  

    group_idxs = get_group_idxs(group_batch)
    loss = torch.zeros(1)[0].to(device)
    for idxs in group_idxs:
        if len(idxs) == 1:
            continue
            
        loss += -1 * torch.cos(RyGT_batch[idxs] - estimated_Ry[idxs]).sum()/batch_size 

    return loss.requires_grad_(True)

def get_group_idxs(group):
    # remove duplicate values
    if type(group) != list: #tensor type
        group = group.tolist()
    values = list(dict.fromkeys(group))
    group_idxs = list()
    for val in values:
        tmp = list()
        for i in range(len(group)):
            if group[i] == val:
                tmp.append(i)
        group_idxs.append(tmp)
    return group_idxs

def GroupLoss(orient_batch, orientGT_batch, confGT_batch, group_batch, device):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]#conf 是在那一個bin上取大

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes].to(device)
    orient_batch = orient_batch[torch.arange(batch_size), indexes].to(device)
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
    return weighted_loss.requires_grad_(True).to(device)

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
    cos_diff = np.cos(PRED - GT)
    return 1 - cos_diff.mean()

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
            
def class2angle(bin_class,residual):
    # angle_per_class=2*torch.pi/float(12)
    angle_per_class=2*np.pi/float(4)
    angle=float(angle_per_class*bin_class)
    angle=angle+residual
    # print(angle)
    return angle

def compute_depth_loss(input, target):
    depth_input, depth_log_variance = input[:, 0:1], input[:, 1:2]
    depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
    depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, target, depth_log_variance)
    return depth_loss

def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5*log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()


def calc_depth_with_alpha_theta(img_W, box_2d, cam_to_img, obj_W, obj_L, alpha, trun=0.0):
    fovx = 2 * np.arctan(img_W / (2 * cam_to_img[0][0]))
    box_W = get_box_size(box_2d)[0] / (1-trun+0.01) #assume truncate related to W only
    visual_W = abs(obj_L*np.cos(alpha)) + abs(obj_W*np.sin(alpha))
    theta_ray = calc_theta_ray(img_W, box_2d, cam_to_img)
    visual_W /= abs(np.cos(theta_ray)) #new added !
    Wview = (visual_W)*(img_W/box_W)
    depth = Wview/2 / np.tan(fovx/2)
    return depth

def angle2class(angle, num_heading_bin):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_heading_bin) #degree:30 radius:0.523
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2) #residual 有正有負

    return class_id, residual_angle

# 0613 added 
def L1_loss_alpha(input, target, alpha, device='cuda:0'):
    weights = [torch.ones(input.shape[0]).to(device), 1+torch.sin(alpha)**2, 1+torch.cos(alpha)**2]
    weights = torch.stack(weights, dim=1).to(device)
    loss = abs(input-target)
    loss *= weights
    return torch.mean(loss)

def box_depth_error_calculation(depth_labels, depth_Calcs, out_range=10):
    class_GT = np.copy(depth_labels) #28742 car
    print(f'num of Car:', class_GT.shape[0])
    class_cal = np.copy(depth_Calcs)
    for depth in [0, 10, 20, 30, 40, 50]:
        class_GT_depth = class_GT[np.logical_and(class_GT >= depth, class_GT < depth+10.)]
        print(f'\tnum of depth {depth}-{depth+10}:', class_GT_depth.shape[0], end=' ')
        if class_GT_depth.shape[0] == 0:
            print()
            continue
        class_cal_depth = class_cal[np.logical_and(class_GT >= depth, class_GT < depth+10.)]
        cal_delta = abs(class_GT_depth - class_cal_depth)
        #cal_delta, _, out_indexes = filter_out_of_range(cal_delta, out_range) # remove prediction out of 10
        print(f'  abs_delta mean:{cal_delta.mean():.3f}m, Out of {out_range}m: {cal_delta[cal_delta>=out_range].shape[0]}')

    # after 60 m
    class_GT_depth = class_GT[class_GT >= 60.]
    print(f'\tnum of depth {depth+10}+:', class_GT_depth.shape[0], end='   ')
    class_cal_depth = class_cal[class_GT >= 60.]
    cal_delta = abs(class_GT_depth - class_cal_depth)
    #cal_delta, _, out_indexes = filter_out_of_range(cal_delta, out_range) # remove prediction out of 10
    print(f'  abs_delta mean:{cal_delta.mean():.3f}m, Out of {out_range}m: {cal_delta[cal_delta>=out_range].shape[0]}')
    
    total = abs(class_GT-class_cal)
    print(f'[Total] mean:{total.mean():.3f}, std:{total.std():.3f}')

#def box_correction(box_2d:list[list[int,int], list[int,int]], H:int, W:int) -> list[list[int,int], list[int,int]]:
def box_correction(box_2d, H, W):
    '''
    [USAGE] fixing out of image boundary 2d box in Elan dataset
    '''
    top_left, btm_right = box_2d
    left, top = top_left
    right, btm = btm_right
    left = min(max(1, left), W-1)
    right = min(max(1, right), W-1)
    top = min(max(1, top), H-1)
    btm = min(max(1, btm), H-1)
    return [[left, top], [right, btm]]

def angle_correction(angle:float) -> float:
    '''
    [USAGE] Transform angle to kitti-format: -pi~pi of .2f
    '''
    if angle > np.pi:
        angle -= 2* np.pi
    elif angle < -1*np.pi:
        angle += 2* np.pi
    return angle

class TrackingObject(object):
    def __init__(self, line):
        self.line = line
        self.class_ = None
        self.box2d = None
        self.dims = None
        self.locs = None
        self.rys = None
        self.id = None
        self.frames = list()
        self.crops = list()
        self.lines = list()
        self.set_info(line)
        
    def set_info(self, line):
        self.lines.append(line)
        elements = line.split()
        for j in range(1, len(elements)):
            elements[j] = float(elements[j])
        top_left = [int(round(elements[4])), int(round(elements[5]))]
        btm_right = [int(round(elements[6])), int(round(elements[7]))]
        self.box2d = np.array([top_left, btm_right])
        self.class_ = elements[0]
        self.alphas = [elements[3]]
        self.dims = [[elements[8], elements[9], elements[10]]]
        self.locs = [[elements[11], elements[12], elements[13]]]
        self.rys = [elements[14]]
        if len(elements) == 16:
            self.id = int(elements[15])
    
    def update_info(self, obj):
        self.box2d = obj.box2d
        self.alphas += obj.alphas
        self.dims += obj.dims
        self.locs += obj.locs
        self.rys += obj.rys
        self.lines.append(obj.line)
        
    def record_frames(self, frame_id):
        self.frames.append(frame_id)

def iou_2d(box1, box2):
    box1 = box1.flatten()
    box2 = box2.flatten()
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    area_sum = abs(area1) + abs(area2)
    
    #計算重疊方形座標
    x1 = max(box1[0], box2[0]) # left
    y1 = max(box1[1], box2[1]) # top
    x2 = min(box1[2], box2[2]) # right
    y2 = min(box1[3], box2[3]) # btm

    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        inter_area = abs((x2-x1)*(y2-y1))
    return inter_area/(area_sum-inter_area)

def keep_same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True