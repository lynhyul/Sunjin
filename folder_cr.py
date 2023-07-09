import pandas as pd

train_df = pd.read_csv("d:/vae/open/open/train_df.csv")
print(len(train_df["label"].unique()))  # type = numpy.array
label_list = train_df["label"].unique().tolist()
len(label_list)  # type = list

import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
for i in range(len(label_list)):  # 레이블 개수 만큼 
    createFolder(f'd:/vae/train2/{label_list[i]}')  # 레이블 폴더를 생성 
    
    
train_folder = os.listdir('d:/vae/train/')
len(train_folder)  # -폴더개수 88 , 사진개수 = 4277개


import shutil

for i in range(len(train_folder) - 88):  # 폴더 생성한것 88개 뺴주는 겁니다.
    
    if train_folder[i][-3:] == "png":   # 확장자가 png면 
        label = train_df.loc[train_df["file_name"] == f"{train_folder[i]}"]["label"].item()
        file_source = f'd:/vae/train/{train_folder[i]}'  # train 폴더에 있는 해당 이미지를
        file_destination = f'd:/vae/train2/{label}/'  # 해당 label 폴더로 이동 
        shutil.move(file_source, file_destination)  # 이동 실행