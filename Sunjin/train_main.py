from keras.applications.efficientnet import *
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import preprocess_input
from keras.applications.efficientnet_v2 import EfficientNetV2M,EfficientNetV2S
import os
from keras.optimizers import *
import random
from tqdm import tqdm
import cv2
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_folder(directory):
    # 폴더 생성 함수
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

class S_EfficientNet() :
    def __init__(self, efficient_ver='B3',pretrained = True) : 
        Eff = [ 'EfficientNetB0',               ## EfficientNet version  
                'EfficientNetB1',
                'EfficientNetB2',
                'EfficientNetB3',
                'EfficientNetB4',
                'EfficientNetB5',
                'EfficientNetB6',
                'EfficientNetB7',
                'EfficientNetV2M',
                 'EfficientNetV2S' ]
        self.efficient_ver = efficient_ver
        Eff_idx = Eff.index(f'EfficientNet{self.efficient_ver}')
        self.efficient_v = eval(Eff[Eff_idx])

        if pretrained == True :             ## pretrained weight 사용 여부
            self.pretrained = 'imagenet'
        elif pretrained == False :
            self.pretrained = None

    def sunjin_efficientnet(self,input_shape = (224,224,3),classes = 46) :              ## SR6동에서 학습하고 있는 Mirero Model
        pretrain_model = self.efficient_v(include_top=False, weights=self.pretrained,input_shape = input_shape)
        x = GlobalAveragePooling2D() (pretrain_model.output)
        x = Dense(classes,activation='softmax') (x)
        model = Model(pretrain_model.input, x)

        return model


    def efficientnet(self,input_shape = (224,224,3),classes = 46, dropout_rate = 0.3) :     ## 사이즈 변경만 가능한 Efficient Model
        pretrain_model = self.efficient_v(include_top=False, weights=self.pretrained,input_shape = input_shape)
        x = GlobalAveragePooling2D() (pretrain_model.output)
        x = Dropout(dropout_rate) (x)
        x = Dense(classes,activation='softmax') (x)
        model = Model(pretrain_model.input, x)
        # model.load_weights('E:/Efficient_test/Result2.h5')
        return model


class Efficient_Train(S_EfficientNet) :
    def __init__(self, 
                 efficient_ver='B3', 
                 pretrained=True, 
                 train_path = '', 
                 val_path = '',
                 result_path = '',
                 efficient_type = 'sunjin'):

        super().__init__(efficient_ver)
        self.train_path = train_path
        self.val_path = val_path
        self.result_path = result_path
        self.efficient_type = efficient_type
        self.pretrained = pretrained


    def train(self,epoch = 100, batch_size = 32, image_size = (224,224) ,dropout_rate = 0.3, optimizer = 'adam', lr = 0.001,early_stop = 50,reduce_lr = 15, reduce_factor = 0.1) :
        idg = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_gen = idg.flow_from_directory(directory= self.train_path,target_size=image_size,batch_size = batch_size)
        val_gen = idg.flow_from_directory(directory = self.val_path, target_size=image_size,batch_size=batch_size)
        create_folder(self.result_path)
        from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

        cp = ModelCheckpoint(self.result_path + 'Result.h5',monitor='val_acc',save_best_only=True,verbose=1)
        es = EarlyStopping(monitor='val_acc', patience= early_stop,verbose= 1)
        rlr = ReduceLROnPlateau(monitor='val_acc',patience= reduce_lr,factor=reduce_factor)

        clist = os.listdir(self.train_path)
        input_shape = image_size + (3,)

        if optimizer == 'adam' :
            optimizer = Adam(lr)
        elif optimizer == 'adamax' :
            optimizer = Adamax(lr)
        elif optimizer == 'sgd' :
            optimizer = SGD(lr)
        elif optimizer == 'rmsprop' :
            optimizer = RMSprop(lr)

        if self.efficient_type == 'sunjin' :
            model = S_EfficientNet(efficient_ver=self.efficient_ver,pretrained=self.pretrained)
            model = model.sunjin_efficientnet(input_shape=input_shape, classes= len(clist))
        else : 
            model = S_EfficientNet(efficient_ver=self.efficient_ver,pretrained=self.pretrained)
            model = model.efficientnet(input_shape=input_shape, classes= len(clist),dropout_rate=dropout_rate)

        model.compile(optimizer= optimizer, loss = 'categorical_crossentropy',metrics=['acc'])
        model.fit(train_gen,batch_size= batch_size, epochs = epoch, validation_data=val_gen, steps_per_epoch= len(train_gen),validation_steps=len(val_gen),
                    callbacks=[cp,es,rlr])
        model.load_weights(self.result_path+'Result.h5')
        model.save(self.result_path+'Result.h5')

class Data_Split() :
    def __init__(self,src_path = '', split_per = 0.2, augmentation = ['flip_v','flip_h','rotate90','rotate180','color_shift'], train_path = '', val_path = '',img_size = 224) :
        self.src_path = src_path
        self.split_per = split_per
        self.augmentation = augmentation
        self.train_path = train_path
        self.val_path = val_path
        self.img_size = img_size

    def Aug_write(self, img = [], path='') :
        flip_v = cv2.flip(img,0)
        flip_h = cv2.flip(img,1)
        rotate90 = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        rotate180 = cv2.rotate(img,cv2.ROTATE_180)
        color_shift = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        bright_ad = img + 15
        bright_sb = img - 10
        for i in range(len(self.augmentation)) :
            if path[-3:] == 'jpg' :
                cv2.imwrite(f'{path[:-4]}_{self.augmentation[i]}.{path[-3:]}',eval(self.augmentation[i]))
            elif path[-4:] == 'jpeg' :
                cv2.imwrite(f'{path[:-5]}_{self.augmentation[i]}.{path[-4:]}',eval(self.augmentation[i]))

    def Random_split(self,val_augcount = 30) :
        clist = os.listdir(self.src_path)
        for cls in clist :
            cls_path = self.src_path + cls + '/'
            cls_train = self.train_path + cls + '/'
            cls_val = self.val_path + cls + '/'
            create_folder(cls_train)
            create_folder(cls_val)
            flist = os.listdir(cls_path)
            random.shuffle(flist)
            for idx in tqdm(range(len(flist)),desc= f'{cls} : ') :
                if flist[idx][-3:] == 'jpg' or flist[idx][-3:] == 'png':
                    img_array = np.fromfile(cls_path+f'{flist[idx]}', np.uint8)
                    img = cv2.imdecode(img_array,1)
                    # img = cv2.imread(cls_path+f'{flist[idx]}')
                    img = cv2.resize(img, (self.img_size,self.img_size))
                    if idx > len(flist) * self.split_per :
                        cv2.imwrite(cls_train+f'{flist[idx]}',img)
                        if len(self.augmentation) >= 1 and self.augmentation != False :
                            self.Aug_write(img = img,path = cls_train+f'{flist[idx]}')
                        else :
                            cv2.imwrite(cls_train+f'{flist[idx]}',img)
                    else :
                        if len(flist) * self.split_per > val_augcount :
                            cv2.imwrite(cls_val+f'{flist[idx]}',img)
                        else :
                            cv2.imwrite(cls_val+f'{flist[idx]}',img)
                            if len(self.augmentation) >= 1 and self.augmentation != False:
                                self.Aug_write(img = img,path = cls_val+f'{flist[idx]}')
                            else :
                                cv2.imwrite(cls_val+f'{flist[idx]}',img)



                  
##Data split ###
ds = Data_Split(src_path= 'E:/xray/',
                train_path='E:/xray_Test/train/',
                val_path='E:/xray_Test/val/',
                split_per = 0.2, 
                augmentation = ['flip_v','flip_h','rotate90','rotate180'],
                img_size = 1024)

# ds.Random_split(val_augcount=300)            ## val data 갯수가 30개 이하인 경우에는 val에도 aug을 적용



# model train ###
model = Efficient_Train(train_path='E:/xray_Test/train/',      ## train path 입력
                        val_path='E:/xray_Test/val/',          ## val path 입력
                        efficient_ver='V2S',                         ## Efficient 모델 선택
                        pretrained=True,                            ## pretrained weight 사용 여부
                        result_path = 'E:/Efficient_test/',         ## 모델 저장 파일 경로
                        efficient_type = 'sunjin')                  ## sunjin custom model or Default model 사용 여부



# model.train(epoch = 30,                    ## 학습 epoch (횟수)
#             batch_size = 6,                ## 한 Step당 몇장을 넣어서 학습 할 것인지에 대한 지표 (gpu성능에 따라 결정)
#             image_size = (1024,1024) ,        ## image_size
#             optimizer = 'adam',             ## optimizer
#             lr = 0.001,                     ## learning_rate
#             early_stop = 15,                ## ealry_stop 횟수 -> 50번이상 acc갱신이 일어나지 않으면 학습을 조기 종료
#             reduce_lr = 15,                 ## learning_rate decay, 15번이상 acc갱신이 일어나지 않으면 lr을 줄여주는 역할
#             reduce_factor = 0.1)            ## lr을 줄일 때 lr에 0.1만큼 곱해주는 기능
