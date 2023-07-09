import IMP
import cv2
import imutils
import os
import numpy as np

path = 'd:/test_image/'
# img = IMP.Tool.Image_Load(f'{path}test4.jpg',1)
img = IMP.Tool.Image_Load(f'{path}가동중 (4).bmp',1)
img = IMP.Tool.Rotate(img,9)
cv2.imwrite(path+'test1.jpg',img)
IMP.Tool.Show(img)


# r,g,b = cv2.split(img)

# h,w = r.shape

# r = np.where(r> 10,)

# # r,th = IMP.Binary.IntensityThresholding(r,100,'inv')
# # r = IMP.Morphology.Closing(r,2,2)


# # img = IMP.Filter.Median(img,3)
# # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # print(img.shape)
# # img = IMP.Tool.Resize(img,1024,840)
# # img = IMP.Tool.Resize(img,1024,1024)
# # ref_img = IMP.Tool.Image_Load(path+'test3.jpg',1)
# # print(ref_img.shape)


# # #mathcing
# # top_left,bottom_right,dst = IMP.Matching.Template_Matching(img,ref_img)
# # dst,clip_image = IMP.Matching.Pattern_Matching(img,ref_img)
# test = 0

# IMP.Tool.Show_Resize(r,w,h)
# IMP.Tool.Show(dst)

# cv2.imwrite(f'{path}result.jpg',dst)
# img = IMP.Tool.Resize(img,480,480)
# img = IMP.Filter.Gausian_Blur(img,5)
# img = cv2.Canny(img, 65, 200)

# contours를 찾아 크기순으로 정렬
# cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


