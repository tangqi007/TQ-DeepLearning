from tensorflow.keras.layers import *
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.models import Model
import tensorflow as tf


config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def block(inputs,number):
    x=Conv2D(number,kernel_size=3,strides=1,padding='same')(inputs)
    x=Conv2D(int(number)//4,kernel_size=3,strides=1,padding='same')(x)
    x=InstanceNormalization(axis=3)(x)
    x=Activation('relu')(x)

    x = Conv2D(number, kernel_size=3, strides=1, padding='same')(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x=add([x,inputs])
    x=Activation('relu')(x)
    return x

def giveup(inputs,number):
    x=UpSampling2D(size=(2,2))(inputs)
    x=Conv2D(number,kernel_size=3,strides=1,padding='same')(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x


def resnet():
    inputs=Input(shape=(128,128,3))
    x1=Conv2D(64,kernel_size=9,strides=1,padding='same')(inputs)
    x1 = InstanceNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1) #128 128 64

    x2=Conv2D(64,kernel_size=7,strides=1,padding='same')(inputs)
    x2 = InstanceNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)  # 128 128 64

    x3 = Conv2D(64, kernel_size=5, strides=1, padding='same')(inputs)
    x3 = InstanceNormalization(axis=3)(x3)
    x3= Activation('relu')(x3)  # 128 128 64

    x=add([x2,x1,x3]) #128 128 64

    x=Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)  # 128 128 64
    v1=block(x,64)

    v2=Conv2D(128, kernel_size=3, strides=2, padding='same')(v1)
    v2 = InstanceNormalization(axis=3)(v2)
    v2 = Activation('relu')(v2)  # 64 64 128
    v2=block(v2,128)

    v3=Conv2D(256, kernel_size=3, strides=2, padding='same')(v2)# 32 32 256
    v3=InstanceNormalization(axis=3)(v3)
    v3=Activation('relu')(v3)
    for _ in range(5):
       v3=block(v3,256)

    v4=Conv2D(256, kernel_size=3, strides=2, padding='same')(v3)
    v4 = InstanceNormalization(axis=3)(v4)
    v4 = Activation('relu')(v4)
    v4=block(v4,256)  # 16 16 256

    v5=Conv2D(256, kernel_size=3, strides=2, padding='same')(v4)
    v5 = InstanceNormalization(axis=3)(v5)
    v5 = Activation('relu')(v5)
    v5=block(v5,256)  # 8 8 256

    up1=giveup(v5,256)
    up1=add([up1,v4])# 16 16 256

    up2=giveup(up1,256)
    up2=add([up2,v3])# 32 32 256

    up3=giveup(up2,128)
    up3=add([up3,v2]) #64 64 128

    up4=giveup(up3,64)
    up4=add([up4,v1])# 128 128 64

    a=Conv2D(256, kernel_size=3, strides=1, padding='same')(up4)
    a = InstanceNormalization(axis=3)(a)
    a = Activation('relu')(a)
    a=Conv2D(128, kernel_size=3, strides=1, padding='same')(a)
    a=block(a,128)
    a=Conv2D(3, kernel_size=7, strides=1, padding='same')(a)
    a=Activation('tanh')(a)

    return Model(inputs,a)









