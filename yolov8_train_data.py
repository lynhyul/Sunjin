import os
import IMP
import cv2
import numpy as np

path = '//193.202.10.241/디지털혁신센터/디지털기술팀/01.5_스마트팩토리/03_식육라벨지검사/05_데이터/'
save_path = "d:/yolo/test/"
# save_path = "d:/yolo/data/06_19/images/"

IMP.Tool.create_folder(save_path)

flist = os.listdir(path)
flist2 = os.listdir(save_path)
day_th = 5
month_th = 7

for file in flist :
    try :
        print(file)
        day = file.split('_')[2]
        month = file.split('_')[1]
        if day[0] == '0' :
            day = day[1]
        day = int(day)
        month = int(month[1])
        if file in flist2 or day < day_th or month < month_th or file in flist2:
            pass
        else :
            fpath = path + file
            img = IMP.Tool.Image_Load(fpath,1)
            if np.mean(img) < 30 :
                pass
            else :
                img = cv2.resize(img,(640,640))
                cv2.imwrite(f'{save_path}{file}',img)
    except :
        pass
