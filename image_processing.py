import IMP
import cv2
import imutils
import os

path = 'd:/test_image/'
file = os.listdir(path)[-1]
# img = IMP.Tool.Image_Load(f'{path}test4.jpg',1)
img = IMP.Tool.Image_Load(path+'img2.jpg',0)
img = cv2.resize(img,None,None,1/5,1/5)
_,th = IMP.Binary.IntensityThresholding(img,80,'otsu')
img,th = IMP.Binary.IntensityThresholding(img,th-20)
IMP.Tool.Show(img)
img = IMP.Morphology.Erosion(img,33,5,3)
ref_img = IMP.Tool.Image_Load(path+'ref.png',1)
# contours를 찾아 크기순으로 정렬
rgbimg,Count,filtered_area = IMP.Countour.blob(img,500,100000,'area')
IMP.Tool.Show(rgbimg)
findCnt = None

