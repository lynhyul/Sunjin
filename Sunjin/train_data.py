import cv2
import numpy as np
import os

def create_folder(directory):
    # 폴더 생성 함수
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


path = 'E:/crol/'

train_path = path + 'train/'
label_path = path + 'label/'

create_folder(train_path)
create_folder(label_path)

flist = os.listdir(path)

for file in flist :
    if file[-3:] == 'jpg' or  file[-3:] == 'png' or file[-4:] == 'jpeg' :
        file_path = path + file 
        save_path1 = train_path + file[:-3] + 'jpg'
        save_path2 = label_path + file[:-3] + 'jpg'
        img = cv2.imread(file_path,1)
        img = cv2.resize(img,(224,224))

        k1, k2, k3 = 0.5, 0.2, 0.0 # 배럴 왜곡
        #k1, k2, k3 = -0.3, 0, 0    # 핀큐션 왜곡

        rows, cols = img.shape[:2]

        # 매핑 배열 생성 ---②
        mapy, mapx = np.indices((rows, cols),dtype=np.float32)

        # 중앙점 좌표로 -1~1 정규화 및 극좌표 변환 ---③
        mapx = 2*mapx/(cols-1)-1
        mapy = 2*mapy/(rows-1)-1
        r, theta = cv2.cartToPolar(mapx, mapy)

        # 방사 왜곡 변영 연산 ---④
        ru = r*(1+k1*(r**2) + k2*(r**4) + k3*(r**6)) 

        # 직교좌표 및 좌상단 기준으로 복원 ---⑤
        mapx, mapy = cv2.polarToCart(ru, theta)
        mapx = ((mapx + 1)*cols-1)/2
        mapy = ((mapy + 1)*rows-1)/2
        # 리매핑 ---⑥
        distored = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

        cv2.imwrite(save_path1,img)
        cv2.imwrite(save_path2,distored)
