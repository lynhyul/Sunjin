import os
import cv2
from train_main import S_EfficientNet
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import preprocess_input
import numpy as np
import pandas as pd

root_path = 'E:/'
model_path = root_path + 'Efficient_test/Result.h5'
test_path = 'E:/xray_Test/test/'
df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
label = []
img_list = []
file_name= []


clist = os.listdir(test_path)

# for cls in clist :
#     file_path = test_path + cls + '/'
#     flist = os.listdir(file_path)
#     for file in flist :
#         img = cv2.imread(file_path+file)
#         img = cv2.cvtColor(cv2.resize(img,(512,512)),cv2.COLOR_BGR2RGB)
#         img_list.append(img)
#         file_name.append(file[:-4])
#         if cls == '0' or cls == '2' : 
#             label.append(1)
#         else :
#             label.append(0)

flist = os.listdir(test_path)
for file in flist :
    img = cv2.imread(test_path+file)
    img = cv2.cvtColor(cv2.resize(img,(1024,1024)),cv2.COLOR_BGR2RGB)
    img_list.append(img)
    file_name.append(file[:-4])
    label.append(0)

img_list = np.array(img_list)
label = np.array(label)

eff = S_EfficientNet(efficient_ver='V2S').sunjin_efficientnet(input_shape = (1024,1024,3),classes = 2)
eff.load_weights(model_path)

predict = eff.predict(img_list,batch_size=1)
predict_num = np.argmax(predict,axis=-1)
predict_score = []
for i in range(len(predict)) :
    predict_score.append(max(predict[i]))
    # if max(predict[i]) < 0.9 :
    #     predict_num[i] = 0
    # else :
    #     pass

    # if predict_num[i] == 0 or predict_num[i] == 2 :
    #     predict_num[i] = 1
    # else :
    #     predict_num[i] = 0

score = 0
ADC_code = []
for i in range(len(predict_num)) :
    ADC_code.append(predict_num[i])
    if label[i] == predict_num[i] :
        score += 1/len(predict_num)

print(score)

df['File_name'] = file_name
df['cls_Code'] = label
df['ADC_Code'] = ADC_code
df['Predict_Score'] = predict_score
df.to_csv(f'{root_path}/Result3.csv',index=False)