import cv2
import numpy as np
import imgaug.augmenters as iaa
import random
import os
import IMP

# 이미지 불러오기
# image = cv2.imread('e:/gt_train/image_1.jpg')

count = 0
count2 = 0
labels = []
file_names = []
result_text = ''
result = 'label2'
image_path = 'd:/ref_image/data/'

flist = os.listdir(image_path)

for file in flist :
    img = IMP.Tool.Image_Load(image_path+file,1)
    # 이미지 증강 기술 정의
    seq = iaa.Sequential([
        # iaa.Affine(rotate=(-10, 10)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.GammaContrast(gamma=(0.5, 2.0)),
        iaa.Sharpen(alpha=(0, 0.5)),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.Sometimes(0.5, iaa.MotionBlur(k=(3, 7))),
    ])

    # 이미지 증강 적용
    image_aug = seq.augment_image(img)

    # 증강된 이미지 저장
    cv2.imwrite(f'D:/ref_image/val_data/'+file, image_aug)
