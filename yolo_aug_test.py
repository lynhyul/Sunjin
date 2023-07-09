import cv2
import numpy as np
import random
import IMP
import os

path = 'D:/yolo/data/06_30/'
# save_path = 'd:/yolo/data/06_27/val/'



def save_text_file(input_file, output_file):
    with open(input_file, 'r') as f:
        text_data = f.read()

    with open(output_file, 'w') as f:
        f.write(text_data)

def augmentation(image_path,label_path) :
    image = IMP.Tool.Image_Load(image_path,1)
    cv2.imwrite(image_path[:-4]+'_copy.jpg',image)
    save_text_file(label_path,label_path[:-4]+'_copy.txt')
    ## Color 변환
    cvt_img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path[:-4]+'_cvt.jpg',cvt_img)
    save_text_file(label_path,label_path[:-4]+'_cvt.txt')
     
    # Apply Gaussian noise
    for i in range(3) :
        mean = 0
        std_dev = random.uniform(0, 0.1)  # 무작위 가우시안 노이즈 표준편차 설정
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        cv2.imwrite(image_path[:-4]+f'_noise{i}.jpg',noisy_image)
        save_text_file(label_path,label_path[:-4]+f'_noise{i}.txt')

    



image_path = path + 'images/'
label_path = path + 'labels/'

# save_image_path = save_path + 'images/'
# save_label_path = save_path + 'labels/'


# IMP.Tool.create_folder(save_image_path)
# IMP.Tool.create_folder(save_label_path)

flist = os.listdir(image_path)

for file in flist :
    image_path = path + 'images/' + file
    label_path = path + 'labels/' + file[:-3] + 'txt'
    augmentation(image_path,label_path)