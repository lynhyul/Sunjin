import IMP
import cv2
import numpy as np
import imutils

file_path = "d:/ref_image//Image_20230503115041566.bmp"
img = IMP.Tool.Image_Load(file_path,1)

def rotate(image, angle, scale):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotated= imutils.rotate_bound(image,-angle)
    # M = cv2.getRotationMatrix2D(center, -angle, scale) # 변환 행렬 메트릭스
    # rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



## Track Bar rotate

def onChange(pos):
    pass

cv2.namedWindow("Trackbar Windows",flags=cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar("angle", "Trackbar Windows", 0, 360, lambda x : x)

cv2.setTrackbarPos("angle", "Trackbar Windows", 0)

try :
    while cv2.waitKey(1) != ord('q'):
        angle = cv2.getTrackbarPos("angle", "Trackbar Windows")
        rotate_image = rotate(img,-angle,1.0)
        cv2.imshow("Trackbar Windows", rotate_image) 
except : 
    cv2.imwrite(file_path,rotate_image)
