import cv2
import numpy as np
import os

flist = os.listdir('E:/xray_Test/train/2/')

path = 'E:/xray_Test/train/2/'

for file in flist :
    img = cv2.imread(path + file)
    bright_ad = img + 15
    bright_sb = img - 15
    cv2.imwrite(path + f'{file[:-4]}_add.jpg',bright_ad)
    cv2.imwrite(path + f'{file[:-4]}_sub.jpg',bright_sb)
