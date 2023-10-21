import torch
import torch.nn.functional as F
import numpy as np
from .Plotting import *
import math

def calc_theta_ray(width, box2d, proj_matrix):#透過跟2d bounding box 中心算出射線角度
    box2d = np.array(box2d).flatten()
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = (box2d[2] + box2d[0]) / 2
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

def get_box_center(box2d):
    box2d = np.array(box2d, dtype=np.int32).flatten()
    return [(box2d[0]+box2d[2])//2, (box2d[1]+box2d[3])//2] #x, y

def get_box_size(box2d):
    box2d = np.array(box2d, dtype=np.int32).flatten()
    width = max(box2d[2]-box2d[0], 1)
    height = max(box2d[3]-box2d[1], 1)
    return width, height

def calc_offset(box2d, d3_location, cam_to_img):
    d2_center = get_box_center(box2d)
    proj_center = project_3d_pt(d3_location, cam_to_img)
    box2d_size = get_box_size(box2d)
    offset_x = proj_center[0] - d2_center[0] #pixel
    offset_y = proj_center[1] - d2_center[1]
    return offset_x, offset_y

#offset ratio -1~1
def calc_center_offset_ratio(box2d, d3_location, cam_to_img):
    d2_center = get_box_center(box2d)
    proj_center = project_3d_pt(d3_location, cam_to_img)
    box2d_size = get_box_size(box2d)
    
    if box2d_size[0] < 10:
        offset_x = 0
    else:
        offset_x = (proj_center[0] - d2_center[0]) / (box2d_size[0]//2)

    if box2d_size[1] < 10:
        offset_y = 0
    else:
        offset_y = (proj_center[1] - d2_center[1]) / (box2d_size[1]//2)
    
    if abs(offset_x) > 1: 
        offset_x = sign(offset_x)*1.0
    if abs(offset_y) > 1:
        offset_y = sign(offset_y)*1.0

    return [offset_x, offset_y]

def offset_to_projected_center(box2d, offset_ratio):
    d2_center = get_box_center(box2d)
    box_size = get_box_size(box2d)
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


# ELAN_method
def calc_theta_corner(width, box2d, proj_matrix):
    img_center = width/2
    offset = min(abs(img_center-box2d[0][0]), abs(img_center-box2d[1][0]))
    theta = np.arctan(offset/proj_matrix[0][0])
    return theta

def calc_depth_by_width_corner(img_W, box2d, cam_to_img, obj_W, obj_L):
    fovx = 2 * np.arctan(img_W / (2 * cam_to_img[0][0]))
    theta = calc_theta_corner(img_W, box2d, cam_to_img)
    # delta / Length = tan() (meters) caused by vision
    delta = obj_L * np.tan(theta)
    # (Wobj+delta) / Wview = box-w / img-w 可以算出Wview
    box_W = get_box_size(box2d)[0]
    Wview = (obj_W+delta)*(img_W/box_W) 
    #print('radian:', theta,', Delta m:', delta, ', Wview m:', Wview)
    depth = Wview/2 / np.tan(fovx/2)
    return depth

# my_method
def calc_depth_with_alpha_theta(img_W, box2d, cam_to_img, obj_W, obj_L, alpha, trun=0.0):
    fovx = 2 * np.arctan(img_W / (2 * cam_to_img[0][0]))
    #box_W = get_box_size(box2d)[0] / (1-trun+0.01) #assume truncate related to W only
    box_W = get_box_size(box2d)[0] / (1-trun) #assume truncate related to W only
    visual_W = abs(obj_L*np.cos(alpha)) + abs(obj_W*np.sin(alpha))
    theta_ray = calc_theta_ray(img_W, box2d, cam_to_img)
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

# 0613 added 0810 updated(3/4*)
def L1_loss_alpha(input, target, alpha, device):
    # [H, W, L]
    weights = [torch.ones(input.shape[0]).to(device), 1+torch.sin(alpha)**2, 1+torch.cos(alpha)**2]
    weights = 3/4*torch.stack(weights, dim=1).to(device)
    loss = abs(input-target)
    loss *= weights
    return torch.mean(loss).to(device)

def box_depth_error_calculation(depth_labels, depth_Calcs, out_range=10):
    class_GT = np.copy(depth_labels) #28742 car
    print(f'num of Car:', class_GT.shape[0])
    class_cal = np.copy(depth_Calcs)
    for depth in [0, 10, 20, 30, 40, 50]:
        class_GT_depth = class_GT[np.logical_and(class_GT >= depth, class_GT < depth+10.)]
        print(f'\t[depth {depth}-{depth+10}] num:', class_GT_depth.shape[0], end=' ')
        if class_GT_depth.shape[0] == 0:
            print()
            continue
        class_cal_depth = class_cal[np.logical_and(class_GT >= depth, class_GT < depth+10.)]
        cal_delta = abs(class_GT_depth - class_cal_depth)
        #cal_delta, _, out_indexes = filter_out_of_range(cal_delta, out_range) # remove prediction out of 10
        print(f'  abs_delta mean:{cal_delta.mean():.3f}m, Out of {out_range}m: {cal_delta[cal_delta>=out_range].shape[0]}', end='')
        print(f'  max_delta :{cal_delta.max():.3f}m')

    # after 60 m
    class_GT_depth = class_GT[class_GT >= 60.]
    print(f'\t[depth {depth+10}+] num:', class_GT_depth.shape[0], end='')
    if class_GT_depth.shape[0] != 0:
        class_cal_depth = class_cal[class_GT >= 60.]
        cal_delta = abs(class_GT_depth - class_cal_depth)
        #cal_delta, _, out_indexes = filter_out_of_range(cal_delta, out_range) # remove prediction out of 10
        print(f'  abs_delta mean:{cal_delta.mean():.3f}m, Out of {out_range}m: {cal_delta[cal_delta>=out_range].shape[0]}', end='')
        print(f'  max_delta :{cal_delta.max():.3f}m')
    else:
        print()
    total = abs(class_GT-class_cal)
    print(f'[Total] mean:{total.mean():.3f}, std:{total.std():.3f}')

#def box_correction(box2d:list[list[int,int], list[int,int]], H:int, W:int) -> list[list[int,int], list[int,int]]:
def box_correction(box2d, H, W):
    '''
    [USAGE] fixing out of image boundary 2d box in Elan dataset
    '''
    top_left, btm_right = box2d
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
    def __init__(self, line, is_flip=False):
        self.line = line
        self.class_ = None
        self.truncated = None
        self.occluded = None
        self.box2d = None
        self.dims = None
        self.locs = None
        self.rys = None
        self.id = None
        self.frames = list()
        self.crops = list()
        self.lines = list()
        self.is_flip = is_flip
        self.set_info(line)
        
    def set_info(self, line):
        self.lines.append(line)
        elements = line.split()
        for j in range(1, len(elements)):
            elements[j] = float(elements[j])
        self.class_ = elements[0]
        self.truncated = [elements[1]]
        self.occluded = [elements[2]]
        if self.is_flip:
            self.alphas = [flip_orient(elements[3])]
        else:
            self.alphas = [elements[3]]
        top_left = [int(round(elements[4])), int(round(elements[5]))]
        btm_right = [int(round(elements[6])), int(round(elements[7]))]
        self.box2d = np.array([top_left, btm_right])
        self.dims = [[elements[8], elements[9], elements[10]]]
        self.locs = [[elements[11], elements[12], elements[13]]]
        self.rys = [elements[14]]
        if len(elements) == 16: #group label
            self.id = int(elements[15])

    def update_info(self, obj):
        self.box2d = obj.box2d
        self.truncated += obj.truncated
        self.occluded += obj.occluded
        self.alphas += obj.alphas
        self.dims += obj.dims
        self.locs += obj.locs
        self.rys += obj.rys
        self.lines.append(obj.line)
    
    def record_frames(self, frame_id):
        self.frames.append(frame_id)
    
    def print_info(self):
        print('Box2d', self.box2d[0], self.box2d[1])
        print('Loc', self.locs)
        print('Dim', self.dims)
        print(f'Alpha:{self.alphas}, Ry:{self.rys}')
        print(f'Trun:{self.truncated}, Occ:{self.occluded}')

def calc_IoU_2d(box1, box2):
    box1 = np.array(box1, dtype=np.int32).flatten()
    box2 = np.array(box2, dtype=np.int32).flatten()
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
        area_overlap = abs((x2-x1)*(y2-y1))

    area_union = area_sum-area_overlap
    return area_overlap/area_union

def keep_same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#0810 add #somewhere wrong
def compute_residual_loss(residual, gt_bin, gt_residual, device, reduction='mean'):
    one_hot = torch.zeros((residual).shape).to(device)
    # make one hot map
    for i in range(gt_bin.shape[0]):
        if gt_bin[i] >= residual.shape[1] or gt_bin[i] < 0:
            print(gt_bin)
        one_hot[i][gt_bin[i]] = 1
    reg_residual = torch.sum(residual * one_hot.to(device), axis=1)
    residual_loss = F.l1_loss(reg_residual, gt_residual, reduction=reduction)
    return residual_loss

def old_residual_loss(orient_residual,truth_bin,truth_residual,device):#truth_residual:B,truth_bin:B,orient_residual:B,12

    one_hot_map=torch.zeros((orient_residual.shape)).to(device).scatter_(dim=1,index=truth_bin.view(-1,1),value=1)#(batch,bin_class)
    heading_res=torch.sum(orient_residual*one_hot_map,dim=1)
    reg_loss=F.l1_loss(heading_res,truth_residual,reduction='mean')
    
    return reg_loss,heading_res

# 1021 updated
def compute_group_loss(REG_alphas, GT_alphas, loss_func):
    GT_groups = get_bin_classes(GT_alphas)
    group_idxs = get_group_idxs(GT_groups) #nested list
    group_loss = torch.tensor(0.)
    for idxs in group_idxs:
        if len(idxs) == 1:
            continue
        reg = REG_alphas[idxs]
        gt = GT_alphas[idxs]
        ratio = reg.shape[0]/REG_alphas.shape[0]
        loss = ratio * loss_func(reg, gt)
        group_loss = torch.add(group_loss, loss)
    return group_loss.requires_grad_(True) # add for 0. tensor

def cos_std_loss(reg, gt):
    return torch.std(torch.cos(reg-gt))

def sin_sin_std_loss(reg, gt):
    return torch.std(torch.sin(reg)- torch.sin(gt))

def compute_compare_group_loss(REG_alphas, GT_alphas):
    GT_groups = get_bin_classes(GT_alphas)
    group_idxs = get_group_idxs(GT_groups) #nested list
    
    group_loss = torch.tensor(0)
    for idxs in group_idxs:
        if len(idxs) == 1:
            continue
        reg = REG_alphas[idxs]
        gt = GT_alphas[idxs]
        ratio = reg.shape[0]/REG_alphas.shape[0]
        loss = ratio * compare_abs_best_loss(reg, gt)
        group_loss = torch.add(group_loss, loss)
    return group_loss.requires_grad_(True)

def compare_abs_best_loss(reg, gt):
    cos_delta = torch.cos(reg-gt)
    best_idx = torch.argmax(cos_delta)
    reg_best_delta = abs(reg - reg[best_idx])
    best_delta_loss = torch.sin(reg_best_delta).mean() #using sin for sin(0)~0
    return best_delta_loss

# TODO remove later, same function as compute_angle_by_bin_residual
def compute_alpha(bin, residual, angle_per_class):
    bin_argmax = torch.max(bin, dim=1)[1]
    residual = residual[torch.arange(len(residual)), bin_argmax]
    alphas = angle_per_class*bin_argmax + residual #mapping bin_class and residual to get alpha
    for i in range(len(alphas)):
        alphas[i] = angle_correction(alphas[i])
    return alphas

# TODO remove later, same function as compute_angle_by_bin_residual
def compute_ry(bin, residual, theta_rays, angle_per_class): 
    bin_argmax = torch.max(bin, dim=1)[1]
    residual = residual[torch.arange(len(residual)), bin_argmax]
    angles = angle_per_class*bin_argmax + residual #mapping bin_class and residual to get alpha
    for i in range(len(angles)):
        angles[i] = angle_correction(angles[i]+theta_rays[i])
    return angles

# 1021 added
def compute_angle_by_bin_residual(bin, residual, angle_per_class, theta_rays=None):
    '''
        theta_rays=None : Alpha
        theta_rays!=None : Ry
    '''
    bin_argmax = torch.max(bin, dim=1)[1]
    residual = residual[torch.arange(len(residual)), bin_argmax] 
    angles = angle_per_class*bin_argmax + residual #mapping bin_class and residual to get alpha
    if theta_rays==None:
        theta_rays = torch.zeros_like(angles)
    for i in range(len(angles)):
        angles[i] = angle_correction(angles[i]+theta_rays[i])
    return angles

# USED IN EXTRA LABELING
def get_bin_classes(array, num_bin=60):
    org_classes = list()
    for value in array:
        bin_class = angle2class(value, num_bin)[0]
        org_classes.append(bin_class)
    
    info_dict = generate_info_dict(array, org_classes, num_bin)
    info_dict = check_neighbor_classes(info_dict, num_bin)
    new_classes = get_new_classes(info_dict, org_classes)
    #diff = 0 if (org_classes == new_classes).all() else 1
    return new_classes#, diff
    
def generate_info_dict(array, org_classes, num_bin):
    tmp = [[] for i in range(num_bin)]
    tmp_dict = dict()
    for class_, value in zip(org_classes, array):
        tmp[class_].append(value)
    for idx, list_ in enumerate(tmp):
        if len(list_)==0:
            continue
        tmp_dict[idx] = {'list': list_, 'mean': sum(list_)/len(list_), 'class': idx}
    return tmp_dict

def check_neighbor_classes(dict_, num_bin):
    keys = sorted(dict_.keys())
    angle_per_class = 2*np.pi / num_bin
    # 0-num_bin-1
    for i in range(len(keys)-1):
        if keys[i] == keys[i+1]-1 and abs(dict_[keys[i]]['mean'] - dict_[keys[i+1]]['mean'] < angle_per_class):
            len_i = len(dict_[keys[i]]['list'])
            len_i_1 = len(dict_[keys[i+1]]['list'])

            class_ = dict_[keys[i]]['class'] if len_i>= len_i_1 else dict_[keys[i+1]]['class']
            dict_[keys[i]]['class'] = class_
            dict_[keys[i+1]]['class'] = class_
    # 0 and num_bin-1 is neighbor 
    if 0 in keys and num_bin-1 in keys and abs(dict_[keys[i]]['mean'] - dict_[keys[i+1]]['mean']) < angle_per_class:
        len_i = len(dict_[keys[i]]['list'])
        len_i_1 = len(dict_[keys[i+1]]['list'])
        class_ = dict_[keys[i]]['class'] if len_i>= len_i_1 else dict_[keys[i+1]]['class']
        dict_[keys[i]]['class'] = class_
        dict_[keys[i+1]]['class'] = class_
    return dict_

def get_new_classes(dict_, classes):
    classes = np.array(classes)
    for key in dict_.keys():
        if key != dict_[key]['class']:
            classes[classes==key] = dict_[key]['class']
    return classes
###

def flip_orient(angle):
    if angle>=0: 
        return round(3.14-angle, 2)
    elif angle<0:
        return round(-3.14-angle, 2)
    
## 0919 added

def loc3d_2_box2d_np(orient, location, dimension, cam_to_img):
    prj_points = []
    R = np.array([[np.cos(orient), 0, np.sin(orient)], [0, 1, 0], [-np.sin(orient), 0, np.cos(orient)]])
    corners = create_corners(dimension, location, R)
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        prj_points.append(point)

    prj_points = np.array(prj_points)
    prj_points_X = prj_points[:,0]
    prj_points_Y = prj_points[:,1]
    prj_box = [min(prj_points_X), min(prj_points_Y), max(prj_points_X), max(prj_points_Y)]
    prj_box = np.array(prj_box, dtype=np.int32)
    return prj_box

def calc_GIoU_2d(box1, box2):
    box1 = np.array(box1, dtype=np.int32).flatten()
    box2 = np.array(box2, dtype=np.int32).flatten()
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
        area_overlap = abs((x2-x1)*(y2-y1))

    area_union = area_sum-area_overlap
    IoU = area_overlap/area_union

    #計算凸型面積 (包住AB的長方形 - union AB)
    x1 = min(box1[0], box2[0]) # left
    y1 = min(box1[1], box2[1]) # top
    x2 = max(box1[2], box2[2]) # right
    y2 = max(box1[3], box2[3]) # btm
    
    if x1 >= x2 or y1 >= y2:
        area_C = 0
    else:
        area_C = abs((x2-x1)*(y2-y1))

    GIoU= IoU - (area_C-area_union)/area_C
    return GIoU

def calc_IoU_loss(box2d, theta_rays, dims, alphas, calib):
    iou_loss = torch.tensor(0.0)
    rys = alphas + theta_rays
    for i in range(len(dims)):
        reg_loc, _ = calc_location(dims[i], calib[i], box2d[i], rys[i], theta_rays[i])
        prj_box2d = loc3d_2_box2d_np(rys[i], reg_loc, dims[i], calib[i])
        #iou_loss += torch.tensor(1 - iou_value) #0919ver. 會和giou_loss大小相同
        #iou_loss += -1 * torch.log(torch.tensor(iou_value)) #https://zhuanlan.zhihu.com/p/359982543
        iou_loss += torch.tensor(1 - calc_IoU_2d(box2d[i], prj_box2d))
        #iou_loss += torch.tensor(1 - calc_GIoU_2d(gt_box2d[i], prj_box2d)) GIoU loss
    return iou_loss / len(dims)

def box2d_area(box):
    if len(box)==2: #[ [left, top], [right, btm] ]
        area = (box[1][0]-box[0][0])*(box[1][1]-box[0][1])
    elif len(box)==4: #[left, top, right, btm]
        area = (box[2]-box[0])*(box[3]-box[1]) #有可能會overflow    
    return area

class IoULoss(torch.nn.Module):
    def __init__(self, weight):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(IoULoss, self).__init__()
        self.weight = weight
#gt_box2d, gt_theta_ray, reg_dims, reg_alphas, calib
    def forward(self, gt_box2d, gt_theta_ray, reg_dims, reg_alphas, calib):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        iou_loss = torch.tensor(0.0)
        reg_ry = reg_alphas + gt_theta_ray
        for i in range(len(reg_dims)):
            reg_loc, _ = calc_location(reg_dims[i], calib[i], gt_box2d[i], reg_ry[i], gt_theta_ray[i])
            prj_box2d = loc3d_2_box2d_np(reg_ry[i], reg_loc, reg_dims[i], calib[i])
            iou_value = calc_IoU_2d(gt_box2d[i], prj_box2d)
            #iou_loss += torch.tensor(1 - iou_value) #0919ver. 會和giou_loss大小相同
            #iou_loss += -1 * torch.log(torch.tensor(iou_value)) #https://zhuanlan.zhihu.com/p/359982543
            iou_loss += F.l1_loss(torch.tensor(1.0), torch.tensor(iou_value)) #0926ver. cause somewhere wrong above (not converge)
        iou_loss /= len(reg_dims)
        return self.weight * iou_loss
    
def init_loss_dict():
    loss_dict = dict()
    # org
    loss_dict['total'] = 0 
    loss_dict['dim'] = 0
    loss_dict['bin'] = 0
    loss_dict['residual'] = 0
    loss_dict['theta'] = 0 # theta = bin+residual
    # mine
    loss_dict['group'] = 0
    loss_dict['C_dim'] = 0
    loss_dict['C_angle'] = 0
    loss_dict['depth'] = 0
    loss_dict['iou'] = 0
    return loss_dict

def loss_dict_add(loss_dict, batch_size, bin, residual, dim, total, group, consist_dim, consist_angle, depth, iou): #.item()
    # org
    loss_dict['total'] += total*batch_size
    loss_dict['dim'] += dim*batch_size
    loss_dict['bin'] += bin*batch_size
    loss_dict['residual'] += residual*batch_size
    loss_dict['theta'] = loss_dict['bin'] + loss_dict['residual']
    # mine
    loss_dict['group'] += group*batch_size
    loss_dict['C_dim'] += consist_dim*batch_size
    loss_dict['C_angle'] += consist_angle*batch_size
    loss_dict['depth'] += depth*batch_size
    loss_dict['iou'] += iou*batch_size
    return loss_dict

def calc_avg_loss(loss_dict, total_num): #len(dataset_train_all)
    for key in loss_dict.keys():
        loss_dict[key] /= total_num
    return loss_dict

def print_epoch_loss(loss_dict, epoch, type='Train'): #len(dataset_train_all)
    print(f'--- epoch {epoch} {type}---', end=' ')
    for key in loss_dict.keys():
        if key == 'group': # for better observation
            print()
            print('\t\t\t', end='')
        print(f'[{key}:{loss_dict[key]:.3f}]', end=', ')
    print()