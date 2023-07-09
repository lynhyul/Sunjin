import cv2
import numpy as np
import os
import random

def trans_label_yolo(img,sx,ex,sy,ey) :
    '''
    sx    # 좌측 최상단 x좌표
    sy    # 좌측 최상단 y좌표
    ex - sx   # Box 가로
    ey - sy   # Box 세로
    '''
    box = [sx,ex,sy,ey]
    img_h,img_w = img.shape[:2]
    dw = 1./img_w     ## 이미지 width
    dh = 1./img_h     ## 이미지 heigh
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x*dw,6)
    w = round(w*dw,6)
    y = round(y*dh,6)
    h = round(h*dh,6)
    return (x,y,w,h)
    

def create_folder(directory):
    # 폴더 생성 함수
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# src_path = 'c:/xray/'
result_path_img = 'c:/xray_gen/image/'
result_path_txt = 'c:/xray_gen/label/'
create_folder(result_path_img)
create_folder(result_path_txt)

folder = 'c:/xray/'
foldlist = os.listdir(folder)

for fold in foldlist :
    src_path = folder + fold + '/'
    flist = os.listdir(src_path)

    for file in flist :
        points = [] 
        count = 0
        src = cv2.imread(src_path+file,1)
        
        # if src.shape[0] != 512 :
        h,w = src.shape[:2]
        centerh = h/2
        centerw = w/2
        randomx = random.randint(centerh-200,centerh+200)
        randomy = random.randint(centerh-200,centerh+200)
        rand_length = random.randint(30,60)
        triger = random.randint(0,1)
        th = random.randint(80,120)
        dice = random.randint(1,4)
        for i in range(1,dice+1) :
            value = i * 100
            sx = randomx
            sy = randomy + value
            ey = sy + rand_length
            if triger == 0 :
                ex = sx + rand_length
            else : 
                ex = sx - rand_length
            
            if src[sy,sx][0] < 242 and src[ey,ex][0] < 242:
                
                cv2.line(src, (sx, sy), (ex, ey),(th,th,th),3) # 두께 5
                if triger == 0 :
                    # cv2.rectangle(src, (sx-10,sy-10),(ex+10,ey+10),(255,0,0),3)
                    x,y,w,h = trans_label_yolo(src,sx-10,ex+10,sy-10,ey+10)
                else :
                    # cv2.rectangle(src, (sx+10,sy-10),(ex-10,ey+10),(255,0,0),3)
                    x,y,w,h = trans_label_yolo(src,sx+10,ex-10,sy-10,ey+10)
                points.append(f'0 {x} {y} {w} {h}\n')
                count += 1
            else :
                pass
        if count > 0 :
            label = open(f"{result_path_txt}/{file[:-4]}.txt", 'w')
            cv2.imwrite(result_path_img+file,src)
            for point in points :
                label.write(point)
            label.close()
        
        

    # cv2.imshow('img',src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()