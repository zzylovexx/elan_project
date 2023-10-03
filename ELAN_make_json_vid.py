import cv2, os, glob


def main():
    data_root = 
    for sub_f in sorted(os.listdir(data_root)):
        sub_folder = os.path.join(data_root, sub_f)
        print(sub_folder)
        os.makedirs(os.path.join(sub_folder, weights_name), exist_ok=True)
        img_save_folder = os.path.join(sub_folder, weights_name, 'img')
        os.makedirs(img_save_folder, exist_ok=True)

        with open(f'{sub_folder}/TEST_images.json') as f:
            img_paths = json.load(f)
        label_json = f'{sub_folder}/TEST_objects.json'
        with open(label_json) as f:
            label_dicts = json.load(f)
        
        result_folder = list()
        result_json = os.path.join(sub_folder, weights_name, 'REG_result.json')


def make_video(folder, img_paths, reg_folder, fps=15):
    video_name = os.path.join(folder, f'{reg_folder}.avi')
    image0 = os.path.join(folder, img_paths[0].replace('images', f'{reg_folder}'))
    frame = cv2.imread(image0)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, fps=fps, frameSize=(width,height))
    for i in range(len(img_paths)):
        path = os.path.join(folder, img_paths[i].replace('images', reg_folder))
        video.write(cv2.imread(path))
    cv2.destroyAllWindows()
    video.release()