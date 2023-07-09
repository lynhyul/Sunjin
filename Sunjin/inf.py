from keras.applications.efficientnet import *
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.callbacks import *
import keras


### bottleneck ##
def CBAR_block(input, num_filters) :            
    x = Conv2D(filters=num_filters, kernel_size=1, padding='same') (input)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)

    x = Conv2D(filters=num_filters, kernel_size=3, padding='same') (x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)

    xd = Conv2D(filters=num_filters, kernel_size =1) (input)
    x = Add() ([x,xd])

    return x

# EfficientNet 버전 호출을 위한 변수
Eff = [ 'EfficientNetB0',
        'EfficientNetB1',
        'EfficientNetB2',
        'EfficientNetB3',
        'EfficientNetB4',
        'EfficientNetB5',
        'EfficientNetB6',
        'EfficientNetB7' ]

# EfficientNet backbone_input 호출을 위한 함수 생성
def get_efficientnet(name='B3', input_shape=(None,None,3),pretrained = True) :
    if pretrained == True :
        weights = 'imagenet'
    elif pretrained == False :
        weights = None
    Eff_idx = Eff.index(f'EfficientNet{name}')
    return eval(Eff[Eff_idx])(include_top = False,
                          weights = weights,
                          input_tensor = None,
                          input_shape = input_shape,
                          pooling = None)

# UNet-Efficient model
def efficient_unet(input_shape = (224,224,3), num_classes=1, name = 'B3') :

    encoder_model = get_efficientnet(name=name, input_shape = input_shape)
    backbone_input = encoder_model.input
    backbone_output = encoder_model.get_layer(name='block7a_project_bn').output

    fn_bottle_neck = backbone_output.shape[-1]
    bottleneck = CBAR_block(backbone_output, fn_bottle_neck)

    c1 = encoder_model.get_layer(name= 'block5c_drop').output
    fn_1 = c1.shape[-1]

    upsamling1 = UpSampling2D() (bottleneck)
    concatenation1 = UpSampling2D() (bottleneck)
    concatenation1 = concatenate([upsamling1, c1], axis=3)

    decoder1 = CBAR_block(concatenation1, fn_1)

    c2 = encoder_model.get_layer(name = 'block3b_drop').output
    fn_2 = c2.shape[-1]
    upsampling2 = UpSampling2D() (decoder1)
    concatenation2 = concatenate([upsampling2, c2], axis=3)
    decoder2 = CBAR_block(concatenation2, fn_2)

    c3 = encoder_model.get_layer(name = 'block2b_drop').output
    fn_3 = c3.shape[-1]
    upsampling3 = UpSampling2D() (decoder2)
    concatenation3 = concatenate([upsampling3, c3], axis=3)
    decoder3 = CBAR_block(concatenation3, fn_3)

    c4 = encoder_model.get_layer(name = 'block1a_project_bn').output
    fn_4 = c4.shape[-1]
    upsampling4 = UpSampling2D() (decoder3)
    concatenation4 = concatenate([upsampling4, c4], axis=3)
    decoder4 = CBAR_block(concatenation4, fn_4)

    fn_5 = fn_4
    upsampling5 = UpSampling2D() (decoder4)
    concatenation5 = concatenate([upsampling5, backbone_input], axis=3)
    decoder5 = CBAR_block(concatenation5, fn_5)

    # if num_classes ==1 or num_classes == 2 :
    #     final_filter_num = 1
    #     final_activation = 'sigmoid'

    # else :
    final_filter_num = num_classes
    final_activation = 'tanh'

    model_output = Conv2D(filters = final_filter_num, kernel_size = 1, activation = final_activation) (decoder5)
    print("output shape", model_output.shape)
    efficient_unet = keras.Model(inputs = backbone_input, outputs = model_output)

    return efficient_unet


model = efficient_unet(input_shape = (224,224,3), num_classes=3, name = 'B3')

model.load_weights('e:/test.h5')

from keras.preprocessing.image import ImageDataGenerator

image_height,image_width =224,224

idg2 = ImageDataGenerator()
test_gen = idg2.flow_from_directory('e:/auto_encoder/image/', 
                                                           shuffle= False, target_size=(image_height, image_width),
                                                           batch_size=32,
                                                           class_mode=None)


predict = model.predict(test_gen)

predict = (predict + 1) * 127.5


import cv2

for idx,image in enumerate(predict[:3]) :
    img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'e:/{idx}.jpg',img)