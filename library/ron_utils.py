import math
from .Plotting import *

def sign(num):
    return 1 if num>=0 else -1

def get_2d_center(d2_box):
    return ((d2_box[0][0]+d2_box[1][0])//2, (d2_box[0][1]+d2_box[1][1])//2)

def get_box_size(d2_box):
    x_size = max(d2_box[1][0]-d2_box[0][0], 1)
    y_size = max(d2_box[1][1]-d2_box[0][1], 1)
    return (x_size, y_size)

# return pixels
def calc_center_offset(d2_box, d3_location, cam_to_img, resize=224):
    d2_center = get_2d_center(d2_box)
    proj_center = project_3d_pt(d3_location, cam_to_img)
    d2_box_size = get_box_size(d2_box)
    #resize factor
    factor_x = resize / d2_box_size[0] # transform.resize to 224
    factor_y = resize / d2_box_size[1] 
    
    offset_x = (proj_center[0] - d2_center[0]) * factor_x
    offset_y = (proj_center[1] - d2_center[1]) * factor_y
    
    # offset out of range
    if abs(offset_x) > resize//2: 
        offset_x = sign(offset_x)*resize//2
    if abs(offset_y) > resize//2:
        offset_y = sign(offset_y)*resize//2
        
    return [math.floor(offset_x), math.floor(offset_y)]

# return ratio between (-1,1)
def calc_center_offset_ratio(d2_box, d3_location, cam_to_img):
    d2_center = get_2d_center(d2_box)
    proj_center = project_3d_pt(d3_location, cam_to_img)
    d2_box_size = get_box_size(d2_box)
    
    offset_x = (proj_center[0] - d2_center[0]) / float(d2_box_size[0])
    offset_y = (proj_center[1] - d2_center[1]) / float(d2_box_size[1])
        
    return [offset_x, offset_y]

def calc_theta_ray(img, box_2d, proj_matrix):#透過跟2d bounding box 中心算出射線角度
    width = img.shape[1]
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