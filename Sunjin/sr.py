from keras.applications.efficientnet import *
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.callbacks import *
import keras

initializer = tf.random_normal_initializer(0.,0.02)
image_height,image_width =256,256
inputs = Input(shape=(image_height,image_width,3))
layer = UpSampling2D((2,2))(inputs)

layer1 = Conv2D(filters=64,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer)
layer1 = LeakyReLU()(layer1)
layer1_ = layer1

layer2 = Conv2D(filters=128,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer1)
layer2_ = BatchNormalization()(layer2)
layer2 = LeakyReLU()(layer2_)

layer3 = Conv2D(filters=256,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer2)
layer3_ = BatchNormalization()(layer3)
layer3 = LeakyReLU()(layer3_)

layer4 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer3)
layer4_ = BatchNormalization()(layer4)
layer4 = LeakyReLU()(layer4_)

layer5 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer4)
layer5_ = BatchNormalization()(layer5)
layer5 = LeakyReLU()(layer5_)

layer6 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer5)
layer6_ = BatchNormalization()(layer6)
layer6 = LeakyReLU()(layer6_)

layer7 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer6)
layer7_ = BatchNormalization()(layer7)
layer7 = LeakyReLU()(layer7_)



layer8 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer7)
layer8_ = BatchNormalization()(layer8)
layer8 = LeakyReLU()(layer8_)



layer9 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer8)
layer9 = BatchNormalization()(layer9)
layer9 = layer9+layer7_
layer9 = Dropout(0.5)(layer9)
layer9 = ReLU()(layer9)

layer10 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer9)
layer10 = BatchNormalization()(layer10)
layer10 = concatenate([layer10,layer6_])
layer10 = Dropout(0.5)(layer10)
layer10 = ReLU()(layer10)

layer11 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer10)
layer11 = BatchNormalization()(layer11)
layer11 = layer11+layer5_
layer11 = Dropout(0.5)(layer11)
layer11 = ReLU()(layer11)
                      
layer12 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer11)
layer12 = BatchNormalization()(layer12)
layer12 = layer12+layer4_
layer12 = ReLU()(layer12)
                      
layer13 = Conv2DTranspose(filters=256,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer12)
layer13 = BatchNormalization()(layer13)
layer13 = layer13+layer3_
layer13 = ReLU()(layer13)

layer14 = Conv2DTranspose(filters=128,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer13)
layer14 = BatchNormalization()(layer14)
layer14 = layer14+layer2_
layer14 = ReLU()(layer14)

# layer15 = Conv2DTranspose(filters=64,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer14)
# layer15 = BatchNormalization()(layer15)
# layer15 = layer15+layer1_
# layer15 = ReLU()(layer15)


outputs = Conv2DTranspose(3,4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')(layer14)

model = Model(inputs=inputs,outputs=outputs)



seed=24
batch_size= 32


img_data_gen_args = dict(rescale = 1-(1/127.5))


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


md = ModelCheckpoint("e:/test.h5",monitor='mae',verbose=1,save_best_only=True)

idg = ImageDataGenerator(**img_data_gen_args,validation_split=0.2)
# idg = ImageDataGenerator(**img_data_gen_args,validation_split=0.2,rotation_range=90,zoom_range=0.1,shear_range=0.1)
idg2 = ImageDataGenerator(**img_data_gen_args,validation_split=0.2)
train_generator = idg.flow_from_directory('e:/auto_encoder/image/', 
                                                           seed=seed, target_size=(image_height, image_width),
                                                           batch_size=batch_size,
                                                           class_mode=None, subset='training')

val_generator = idg2.flow_from_directory('e:/auto_encoder/image/', 
                                                           seed=seed, target_size=(image_height, image_width),
                                                           batch_size=batch_size,
                                                           class_mode=None,subset='validation')

mask_gen = idg.flow_from_directory('e:/auto_encoder/label/', 
                                                           seed=seed, target_size=(image_height, image_width),
                                                           batch_size=batch_size,
                                                           class_mode=None, subset='training')

val_mask_gen = idg2.flow_from_directory('e:/auto_encoder/label/', 
                                                           seed=seed, target_size=(image_height, image_width),
                                                           batch_size=batch_size,
                                                           class_mode=None, subset='validation')

steps_per_epoch = len(train_generator)

train_generator = zip(train_generator,mask_gen)
val_generator = zip(val_generator,val_mask_gen)



model.compile(optimizer="adam", loss='mae', metrics=['accuracy'])     

history = model.fit(train_generator, epochs=200,steps_per_epoch=steps_per_epoch,validation_data=val_generator,callbacks=[md],validation_steps=steps_per_epoch)


# model.summary()
