import IMP
import cv2
import imutils
import os
import time

path = 'd:/test_image/'
file = os.listdir(path)[-1]
img = IMP.Tool.Image_Load(path+'test3.png',0)
# img = IMP.Tool.Resize(img, 1024,1024)
ref_img = IMP.Tool.Image_Load(path+'ref.png',0)

# #mathcing
start = time.time()
clip_image = IMP.Matching.pattern_match_rotation(img,ref_img,th=15,gamma=30)
# clip_image = IMP.Matching.Pattern_Matching(img,ref_img)

test = 0
end = time.time()

t_time = abs(round(start - end,2))
print(t_time)

# IMP.Tool.Show_Resize(img,img.shape[1],img.shape[0])
IMP.Tool.Show(img)
IMP.Tool.Show(clip_image)
findCnt = None

