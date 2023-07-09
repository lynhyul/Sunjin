from IMP import *
import cv2
import numpy as np
from NAS_API import NAS_Api
import threading
import time

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2448)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)


nas = NAS_Api(username='sehwang',
                password='tjddms123!'
                ,nas_ip='115.94.24.54',
                nas_port='5003',
                src_save=True)

def grab() :
    count = 0
    while True :
        # ret,frame = capture.read()
        # src = frame
        src1 = Tool.Image_Load('d:/test/test2.jpg',1)
        src = Tool.RGBToGray(src1)
        src = Tool.Resize(src,2048,2448)
        th_image,th = Binary.IntensityThresholding(src,170)
        th_image = Morphology.Opening(th_image,15*4,5*4,3)
        result_img,Count,filtered_area = Countour.blob(th_image,300*4,5000000,'w')
        # result_img2,Count2,filtered_area2 = Countour.blob(th_image2,300*4,5000000,'w')
        print(filtered_area)
        if Count == 2 :
            filter = abs(filtered_area[1][2] - filtered_area[0][2])+abs(filtered_area[1][3] - filtered_area[0][3])
            if  filter < 100 :
                count += 1
                cv2.imwrite(f'd:/test3/{count}.jpg',src1)
                time.sleep(1)
            
def data_api() :          
    nas.Upload(src_path='d:/test3/',
                nas_path='/디지털혁신센터/디지털기술팀/01.5_스마트팩토리/03_식육라벨지검사/05_데이터/')

if __name__ == '__main__' :
    t1 = threading.Thread(target=grab)
    t2 = threading.Thread(target=data_api)
    
    t1.start()
    t2.start()
    
    # print("ok")
    # Tool.Show_Resize(src,512,512)
    # Tool.Show_Resize(result_img,512,512)

# path1 = 'd:/test1.jpg'
# path2 = 'd:/test2.jpg'

# img1 = Tool.Resize(Tool.Image_Load(path1,0),2448 ,2048)
# img2 = Tool.Resize(Tool.Image_Load(path2,0),2448,2048)

# src2 = Tool.Resize(Tool.Image_Load(path1,1),2448,2048)
# src = Tool.Resize(Tool.Image_Load(path2,1),2448,2048)



# th_image2,th = Binary.IntensityThresholding(img1,170)
# th_image2 = Morphology.Opening(th_image2,15*4,5*4,3)

# th_image,th = Binary.IntensityThresholding(img2,th+60)

# Tool.Show(th_image)


# Tool.Show_Resize(src2,512,512)
# Tool.Show_Resize(result_img2,512,512)





