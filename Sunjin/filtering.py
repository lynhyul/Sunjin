import cv2
import numpy as np
import os

path = 'E:/xray/1/'
flist = os.listdir(path)

for file in flist :
    img = cv2.imread(f'{path}{file}',0)
    img = cv2.resize(img,(224,224))
    h,w = img.shape
    per = (h*w /100) * 70
    if np.count_nonzero(img > 242) > per :
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        os.remove(f'{path}{file}')
