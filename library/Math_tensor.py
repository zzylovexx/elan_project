import torch
import torch.nn.functional as F
import numpy as np

# This KEEP grad! (no using torch.linalg.lstsq) https://github.com/pytorch/pytorch/issues/27036#issuecomment-546514208
class LeastSquares:
    def __init__(self, device):
        self.device = device
    
    def lstq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        A = A.to(self.device)
        Y = Y.to(self.device)
        # Assuming A to be full column rank
        cols = A.shape[1]
        if cols == torch.linalg.matrix_rank(A):
            q, r = torch.linalg.qr(A, 'reduced')
            q = q.to(self.device)
            r = r.to(self.device)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols, dtype=torch.float, device=self.device)
            Y_dash = A.permute(1, 0) @ Y
            x = self.lstq(A_dash, Y_dash)
        return x

def calc_IoU_2d_tensor(box1, box2):
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

def calc_IoU_loss_tensor(gt_box2d, gt_theta_ray, reg_dims, reg_alphas, calib, device):
    iou_list = list()
    reg_ry = reg_alphas + gt_theta_ray
    for i in range(len(reg_dims)):
        reg_loc, _ = calc_location_tensor(reg_dims[i], calib[i], gt_box2d[i], reg_ry[i], gt_theta_ray[i], device)
        calib_tensor = torch.tensor(calib[i], dtype=torch.float)
        prj_box2d = loc3d_2_box2d_tensor(reg_ry[i], reg_loc, reg_dims[i], calib_tensor, device)
        iou_list.append(calc_IoU_2d_tensor(gt_box2d[i].to(device), prj_box2d))
    #iou_loss = -1 * torch.log(torch.tensor(iou_value)) #https://zhuanlan.zhihu.com/p/359982543
    iou_list = torch.stack(iou_list)
    best_iou = torch.ones_like(iou_list).to(device)
    iou_loss = F.l1_loss(iou_list, best_iou, reduction='mean') #0926ver. cause somewhere wrong above (not converge)
    return iou_loss

def project_3d_pt_tensor(pt, cam_to_img, device):
    one = torch.tensor([1]).to(device)
    point = torch.cat((pt, one))
    point = torch.matmul(cam_to_img.to(device), point)
    point = point[:2]/point[2]
    return point

def loc3d_2_box2d_tensor(orient, location, dimension, cam_to_img, device):
    prj_points = []
    R = torch.FloatTensor([[torch.cos(orient), 0, torch.sin(orient)], [0, 1, 0], [-torch.sin(orient), 0, torch.cos(orient)]]) 
    corners = create_corners_tensor(dimension, location, R, device)
    for corner in corners:
        point = project_3d_pt_tensor(corner, cam_to_img, device)
        prj_points.append(point)

    prj_points = torch.stack(prj_points)
    prj_points_X = prj_points[:,0]
    prj_points_Y = prj_points[:,1]
    prj_box = torch.stack([min(prj_points_X), min(prj_points_Y), max(prj_points_X), max(prj_points_Y)])
    return prj_box

def create_corners_tensor(dimension, location, R, device):
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = torch.FloatTensor([x_corners, y_corners, z_corners]).to(device)
    R = torch.FloatTensor(R).to(device)
    # rotate with R 
    corners = torch.matmul(R, corners)
    # shift with location 
    for i, loc in enumerate(location):
        corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append(torch.stack([corners[0][i], corners[1][i], corners[2][i]]))
    return torch.stack(final_corners)

def calc_location_tensor(dimension, proj_matrix, box2d, alpha, theta_ray, device):
    # no grad in Rotation matrix (TODO)
    ty = alpha + theta_ray
    R = torch.FloatTensor([[torch.cos(ty), 0, torch.sin(ty)], [0, 1, 0], [-torch.sin(ty), 0, torch.cos(ty)]]) 
    ls = LeastSquares(device)
    proj_matrix = torch.FloatTensor(proj_matrix)
    
    # get the point constraints
    constraints = []
    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    # below is very much based on trial and error

    # based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if alpha < torch.deg2rad(torch.tensor(92.)) and alpha > torch.deg2rad(torch.tensor(88.)):
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif alpha < torch.deg2rad(torch.tensor(-88.)) and alpha > torch.deg2rad(torch.tensor(-92.)):
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif alpha < torch.deg2rad(torch.tensor(90.)) and alpha > -torch.deg2rad(torch.tensor(90.)):
        left_mult = -1
        right_mult = 1

    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1
    if alpha > 0:
        switch_mult = 1

    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1,1):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-1,1):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, dy, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)
    

    # create pre M (the term with I and the R*X)
    pre_M = torch.eye(4)
    # 1's down diagonal
    best_loc = None
    best_error = None
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
    
        # each corner
        Xa = torch.stack(constraint[0], axis=0).view(-1,1)
        Xb = torch.stack(constraint[1], axis=0).view(-1,1)
        Xc = torch.stack(constraint[2], axis=0).view(-1,1)
        Xd = torch.stack(constraint[3], axis=0).view(-1,1)

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = torch.clone(pre_M)
        Mb = torch.clone(pre_M)
        Mc = torch.clone(pre_M)
        Md = torch.clone(pre_M)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = torch.zeros([4,3], dtype=torch.float32)
        b = torch.zeros([4,1])

        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = torch.matmul(R, X)
            M[:3,3] = RX.reshape(3)

            M = torch.matmul(proj_matrix, M)

            A[row, :] = M[index,:3] - box2d[row] * M[2,:3]
            b[row] = box2d[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        #loc, _, rank, s = torch.linalg.lstsq(A, b, rcond=None) # error is empty
        loc = ls.lstq(A, b, 0.0010)
        # for compare error value
        _, error, _, _ = np.linalg.lstsq(A.detach().numpy(), b.detach().numpy(), rcond=None)

        # found a better estimation
        if best_error is None :
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array
            
        elif error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    best_loc = torch.stack([best_loc[0][0], best_loc[1][0], best_loc[2][0]])
    return best_loc, best_X

def get_box_size_tensor(box2d):
    width = box2d[:,2]-box2d[:,0]
    height = box2d[:,3]-box2d[:,1]
    return width, height

def calc_theta_ray_tensor(img_W, box2d, calib):
    box_center = (box2d[:,2] + box2d[:,0]) / 2
    dx = box_center - (img_W / 2)
    theta_ray = torch.arctan((dx/calib[:,0,0]))
    return theta_ray

#obj_W, obj_L = reg_dims[:,1], reg_dims[:,2]
def calc_depth_with_alpha_theta_tensor(img_W, box2d, calib, obj_W, obj_L, alpha, trun, device):
    box_W = get_box_size_tensor(box2d)[0]
    not_trun = torch.ones_like(trun, dtype=torch.float16) - trun #1-trun
    box_W = torch.div(box_W, not_trun).to(device) # trun assume 0?
    visual_W = abs(obj_L*torch.cos(alpha)) + abs(obj_W*torch.sin(alpha))
    fovx = 2 * torch.arctan(img_W / (2 * calib[:,0,0])).to(device)
    theta_ray = calc_theta_ray_tensor(img_W, box2d, calib).to(device)
    visual_W = torch.div(visual_W, abs(torch.cos(theta_ray)))
    Wview = visual_W*img_W.to(device) / box_W
    depth = Wview/2 / torch.tan(fovx/2)
    return depth.to(device)