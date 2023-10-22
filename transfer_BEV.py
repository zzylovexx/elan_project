import glob, os, math, cv2
import numpy as np
import matplotlib.pyplot as plt

# BEV FUNCs
def init_bev(scale=10, bev_size=(720, 360), dis_interval=5, thickness=2):
    x_offset = bev_size[1] // 2 #圓心位置 (車子位置)
    image_bev = np.zeros((bev_size[0], bev_size[1], 3), np.uint8)
    for i in range(1, 20): #黑底白圈
        cv2.circle(image_bev, (x_offset, bev_size[0]), scale*i*dis_interval, (255, 255, 255), thickness)
    return image_bev

def rotate_point(x, y, center_x, center_y, theta):
    x = x - center_x
    y = y - center_y
    nx = int(center_x + x * math.cos(theta) - y * math.sin(theta))
    ny = int(center_y + x * math.sin(theta) + y * math.cos(theta))
    return nx, ny

def draw_bev_rect(image, rect, thickness=2):
    center_x = rect[0]
    center_y = rect[1]
    w = rect[2]
    h = rect[3]
    theta = rect[4]
    x1 = center_x - 0.5 * w 
    x2 = center_x + 0.5 * w 
    y1 = center_y - 0.5 * h 
    y2 = center_y + 0.5 * h 

    point_list = []
    point_list.append(rotate_point(x1, y1, center_x, center_y, theta))
    point_list.append(rotate_point(x1, y2, center_x, center_y, theta))
    point_list.append(rotate_point(x2, y2, center_x, center_y, theta))
    point_list.append(rotate_point(x2, y1, center_x, center_y, theta))

    red = (255, 0, 0)
    blue = (0, 0, 255)
    cv2.line(image, point_list[0], point_list[1], blue, thickness)
    cv2.line(image, point_list[1], point_list[2], blue, thickness)
    cv2.line(image, point_list[2], point_list[3], red, thickness*2) #紅色表示車頭朝向
    cv2.line(image, point_list[3], point_list[0], blue, thickness)

def get_bev_rect(location, dimension, orient, scale, bev_size=(720, 360)):
    x3d, _, z3d = location
    _, w3d, l3d = dimension
    x_offset = bev_size[1] // 2
    bev_rect = [0, 0, 0, 0, 0]
    bev_rect[0] = x3d * scale + x_offset
    bev_rect[1] = bev_size[0] - z3d * scale
    bev_rect[2] = l3d * scale
    bev_rect[3] = w3d * scale
    bev_rect[4] = orient
    return bev_rect

# IMAGES to VIDEO
def make_video(img_folder, vid_name, fps=15):
    video_name = os.path.join(img_folder, f'{vid_name}.avi')
    images = sorted(glob.glob(f'{img_folder}/*.png'))
    height, width, _ = cv2.imread(images[0]).shape
    video = cv2.VideoWriter(video_name, 0, fps=fps, frameSize=(width,height))
    for i in range(len(images)):
        video.write(cv2.imread(images[i]))
    cv2.destroyAllWindows()
    video.release()   

def main():
    data_root = 'Elan_3d_box'
    images = sorted(glob.glob(f'{data_root}/image_2/*.png'))
    labels = sorted(glob.glob(f'{data_root}/label_2/*.txt'))
    BEV_image_folder = f'{data_root}/image_bev'
    os.makedirs(BEV_image_folder, exist_ok=True)
    H, _, _= cv2.imread(images[0]).shape
    BEV_SIZE=(H, H//2+100) # 2:1方便看, 第一個值是為了concat img, 第二個可以是any value
    SCALE = 6 # 越大會讓可視的最遠距離越小)
    INTERVAL = 10 # 一個白圈表示的距離
    FPS = 5 # video fps
    for i in range(len(images)):
        save_path = os.path.join(BEV_image_folder, f'{i:06d}.png')
        lines = open(labels[i]).readlines()
        img = cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2RGB)
        img_bev = init_bev(scale=SCALE, bev_size=BEV_SIZE, dis_interval=INTERVAL, thickness=4)
        for line in lines:
            line = line.split()
            if line[0].lower() == 'dontcare':
                continue
            for j in range(1,15):
                line[j] = float(line[j])

            box2d = [[int(line[4]), int(line[5])], [int(line[6]), int(line[7])]]
            dim = [line[8], line[9], line[10]]
            loc = [line[11], line[12], line[13]]
            global_orient = float(line[14])
            bev_rect = get_bev_rect(loc, dim, global_orient, scale=SCALE, bev_size=BEV_SIZE)
            draw_bev_rect(img_bev, bev_rect)
            cv2.rectangle(img, box2d[0], box2d[1], (255,0,0), 2)
        concat_result = np.concatenate((img, img_bev), axis=1)
        plt.imsave(save_path, concat_result)
        #plt.imshow(concat_result)
        #print(save_path)
    make_video(img_folder=BEV_image_folder, vid_name='BEV', fps=5)

if __name__=='__main__':
    main()