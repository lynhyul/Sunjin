
import json
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import cv2
import IMP
import pandas as pd


# label_paths = ["D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/1. Training/라벨링데이터/TL1/result/medicine/annotations/","D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/1. Training/라벨링데이터/TL2/result/cosmetics/annotations/"]
# image_paths = ['D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/1. Training/원천데이터/TS1/result/medicine/images/','D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/1. Training/원천데이터/TS2/result/cosmetics/images/']

label_paths = ["D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/2. Validation/라벨링데이터/VL1/result/medicine/annotations/","D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/2. Validation/라벨링데이터/VL1/result/cosmetics/annotations/"]
image_paths = ['D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/2. Validation/원천데이터/VS1/result/medicine/images/','D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/2. Validation/원천데이터/VS1/result/cosmetics/images/']

import json
import pandas as pd


count1 = 0
count2 = 0
labels = []
file_names = []
result_text = ''
result = 'result_val'
f = open(f"d:/{result}_gt.txt",'w',encoding='UTF8')

flist2 = os.listdir(f"d:/{result}/")


for i in range(2) :
    label_path = label_paths[i]
    image_path = image_paths[i]
    flist = os.listdir(label_path)
        
    for fi in flist[:4000] :
        file_path = label_path + fi
        # json파일 읽어오기
        file = json.load(open(file_path,encoding='UTF8'))
        file_name = file['images'][0]['name']  
        print(file_name)
        annotations = file['annotations'][0]['bbox']
        polygons = file['annotations'][0]['polygons']

        try :
            for idx,annotation in enumerate(annotations) :
                count1 += 1
                save_name = f'image_{count1}.jpg'
                if save_name in flist2 :
                    pass 
                else :
                    img = IMP.Tool.Image_Load(image_path+file_name,1)
                    sx = int(annotation['x'])
                    sy = int(annotation['y'])
                    ex = sx+int(annotation['width'])
                    ey = sy+int(annotation['height'])
                    img = img[sy:ey,sx:ex]
                    cv2.imwrite(f'd:/{result}/{save_name}',img)
                polygon= polygons[idx]
                text = polygon['text']
                result_text = f'd:/{result}/image_{count1}.jpg\t{text}\n'
                f.write(result_text)        
        except :
            pass

     

# label_paths = ["D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/2. Validation/라벨링데이터/VL1/result/medicine/annotations/","D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/2. Validation/라벨링데이터/VL1/result/cosmetics/annotations/"]
# image_paths = ['D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/2. Validation/원천데이터/VS1/result/medicine/images/','D:/056.의약품, 화장품 패키징 OCR 데이터/01.데이터/2. Validation/원천데이터/VS1/result/cosmetics/images/']

# import json
# import pandas as pd


# count1 = 0
# count2 = 0
# labels = []
# file_names = []
# result = 'result_val'
# f1 = open(f"d:/{result}_gt.txt",'w',encoding='UTF8')

# for i in range(2) :
#     label_path = label_paths[i]
#     image_path = image_paths[i]
#     flist = os.listdir(label_path)

#     for fi in flist :
#         file_path = label_path + fi
#         # json파일 읽어오기
#         file = json.load(open(file_path,encoding='UTF8'))
#         file_name = file['images'][0]['name']
#         print(file_name)
#         annotations = file['annotations'][0]['bbox']
#         polygons = file['annotations'][0]['polygons']


#         try :
#             for idx,annotation in enumerate(annotations) :
#                 count1 += 1
#                 save_name = f'image_{count1}.jpg'
#                 img = IMP.Tool.Image_Load(image_path+file_name,1)
#                 sx = int(annotation['x'])
#                 sy = int(annotation['y'])
#                 ex = sx+int(annotation['width'])
#                 ey = sy+int(annotation['height'])
#                 img = img[sy:ey,sx:ex]
#                 cv2.imwrite(f'd:/{result}/{save_name}',img)
#                 polygon= polygons[idx]
#                 text = polygon['text']
#                 result_text = f'd:/{result}/{save_name}\t{text}\n'
#                 f1.write(result_text)        
#         except :
#             pass         


