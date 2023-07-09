import cv2
import numpy as np
import os

def create_folder(directory):
    # 폴더 생성 함수
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def Aug_write(img = [], path='',augmentation='') :
    flip_v = cv2.flip(img,0)
    flip_h = cv2.flip(img,1)
    rotate90 = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    rotate180 = cv2.rotate(img,cv2.ROTATE_180)
    color_shift = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    bright_ad = img + 15
    bright_sb = img - 15
    for i in range(len(augmentation)) :
        if path[-3:] == 'jpg' :
            cv2.imwrite(f'{path[:-4]}_{augmentation[i]}.{path[-3:]}',eval(augmentation[i]))
        elif path[-4:] == 'jpeg' :
            cv2.imwrite(f'{path[:-5]}_{augmentation[i]}.{path[-4:]}',eval(augmentation[i]))



target_path = 'e:/xray/0/'
flist = os.listdir(target_path)

for file in flist :
    img = cv2.imread(f'{target_path}{file}',1)
    img = cv2.resize(img,(512,512))
    # Aug_write(img,target_path+file,['flip_v','flip_h','rotate90','rotate180'])
    Aug_write(img,target_path+file,['bright_ad','bright_sb'])

