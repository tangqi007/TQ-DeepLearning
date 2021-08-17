import keras.backend as k
from keras.layers import *
from keras import optimizers
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from keras.applications import VGG19
from data_load import *
from keras.models import Model
import tensorflow as tf
import numpy as np
import glob
from img_resize import *
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
'''主干网络和预测部分'''
class TQGAN():
    def __init__(self):
        #预先定义
        self.shape_train=(256,256,6)
        self.shape_test=(256,256,3)
        self.optimizer=optimizers.Adam(learning_rate=0.01)
        #建立生成模型
        self.generator=self.generator()
        #建立vgg模型
        self.vgg19=self.vgg19()
        self.vgg19.compile(loss='mse',optimizer=self.optimizer)
        self.vgg19.trainable=False
        #建立判别模型
        self.cnn_model=self.cnn_model()
        self.cnn_model.trainable=False
        self.cnn_model.compile(loss=['binary_crossentropy'],optimizer=self.optimizer,
                               metrics=['acc'])
        #建立混合模型
        data_inputs=Input(self.shape_train)
        g_img=self.generator(data_inputs)
        vgg_out=self.vgg19(g_img)
        cnn_out=self.cnn_model(g_img)
        self.end_model=Model(data_inputs,[cnn_out,vgg_out])
        self.end_model.compile(loss=['binary_crossentropy','mse'],
                              loss_weights=[1,0.9],
                               optimizer=self.optimizer)
        self.end_model.summary()

    def vgg19(self):
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=self.shape_test)
        end_out = [vgg.layers[9].output]
        return Model(vgg.input, end_out)
    def generator(self):#生成器
        def covn(inputs,number,size,strides,bn=False):
            x=Conv2D(filters=number,kernel_size=size,strides=strides,padding='same')(inputs)
            x=LeakyReLU(alpha=0.2)(x)
            if bn:
                x=BatchNormalization()(x)
            return x
        def upcovn(inputs,number,size):
            b=Conv2DTranspose(filters=number,kernel_size=size,strides=2,padding='same')(inputs)
            b=LeakyReLU(0.2)(b)
            b=BatchNormalization()(b)
            return b
        def block(inputs,number):
            d = Conv2D(number, kernel_size=3, strides=1, padding='same')(inputs)
            d = BatchNormalization()(d)
            d = ReLU()(d)
            d = Conv2D(number, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization()(d)
            d = Add()([inputs, d])
            return d
        def avg(inputs,size):
            f=AveragePooling2D(pool_size=(size,size),strides=size)(inputs)
            return f
        inputs=Input(shape=self.shape_train)#256 256 6
        g1=Conv2D(32,kernel_size=7,strides=1,padding='same')(inputs)#256 256 32
        g2=covn(g1,32,5,1,bn=True)#256 256 32
        g2 = covn(g2, 32, 5, 1, bn=True)
        g3=block(g2,32)#256 256 32
        g3 = block(g3, 32)  # 256 256 32
        g4=covn(g3,64,5,2,bn=True)#128 128 64
        g4 = covn(g4, 64, 5, 1, bn=True)
        g5=block(g4,64)#128 128 64
        g5 = block(g5, 64)  # 128 128 64
        g6=covn(g5,128,3,1,bn=True)#128 128 128
        g6 = covn(g6, 128, 3, 1, bn=True)  # 128 128 128
        g7=block(g6,128)#128 128 128
        g7 = block(g7, 128)  # 128 128 128
        g8=covn(g7,256,3,2,bn=True)#64 64 256
        g8 = covn(g8, 256, 3, 1, bn=True)  # 64 64 256
        g9=block(g8,256)#64 64 256
        g10=covn(g9,512,3,2,bn=True)#32 32 512
        g10 = covn(g10, 512, 3, 1, bn=True)  # 32 32 512
        g11=block(g10,512)#32 32 512
        g11 = block(g11, 512)  # 32 32 512
        f1=avg(g11,2)#16 16 512
        f2=avg(g11,4)#8 8 512
        f3=avg(g11,8)#4 4 512
        f4=avg(g11,16)#2 2 512
        h1=covn(f1,128,1,1,bn=True)#16 16 128
        h2=covn(f2,128,1,1,bn=True)#8 8 128
        h3=covn(f3,128,1,1,bn=True)#4 4 128
        h4=covn(f4,128,1,1,bn=True)#2 2 128
        up1=upcovn(h4,128,3)
        b1=concatenate([up1,h3])#4 4 256
        b1=covn(b1,128,3,1,bn=True)
        up2=upcovn(b1,128,3)
        b2=concatenate([up2,h2])#8 8 256
        b2=covn(b2,128,3,1,bn=True)
        up3=upcovn(b2,128,3)
        b3=concatenate([up3,h1])#16 16 256
        b3=covn(b3,128,3,1,bn=True)
        up4=upcovn(b3,512,5)#32 32 512
        b4=concatenate([up4,g11])#32 32 1024
        b4=covn(b4,256,3,1,bn=True)#32 32 256
        up5=upcovn(b4,256,3)
        b5=concatenate([up5,g9])#64 64 512
        b5=covn(b5,256,3,1,bn=True)#64 64 256
        up6=upcovn(b5,128,3)#128 128 128
        b6=concatenate([up6,g7])#128 128 256
        b6=covn(b6,128,3,1,bn=True)#128 128 128
        up7=upcovn(b6,32,3)#256 256 32
        b7=concatenate([up7,g3])#256 256 64
        n=covn(b7,32,3,1,bn=True)
        n=covn(n,16,1,1,bn=True)
        n=Conv2D(3,kernel_size=1,strides=1,padding='same',activation='tanh')(n)
        return Model(inputs,n)
    def cnn_model(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d =LeakyReLU(alpha=0.2)(d)
            if bn:
                    d = BatchNormalization(momentum=0.8)(d)
            return d
        d0 = Input(shape=self.shape_test)

        d1 = d_block(d0, 64, bn=False)
        d1=d_block(d1,64)
        d2 = d_block(d1, 64, strides=2)
        d3 = d_block(d2, 128)
        d4 = d_block(d3, 128, strides=2)
        d5 = d_block(d4, 256)
        d6 = d_block(d5, 256, strides=2)
        d7 = d_block(d6, 512)
        d8 = d_block(d7, 512, strides=2)
        d9 = Dense(64 * 16)(d8)
        d10 =LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
        return Model(d0,validity) #16 16 1
    def scheduler(self, models, epoch):
            # 学习率下降
        if epoch %5000 == 0 and epoch != 0:
            for model in models:
                lr = k.get_value(model.optimizer.lr)
                k.set_value(model.optimizer.lr, lr * 0.5)
                print("lr changed to {}".format(lr * 0.5))
    def train(self,epochs):
        for epoch in range(epochs+1):
            self.scheduler([self.cnn_model,self.end_model],epoch)
            data1,data2,data3,data4,taget1,taget2=train_load()
            data1=loard(data1)
            data2=loard(data2)
            data3 = loard(data3)
            data4 = loard(data4)
            trains_count=cont(data1,data2,data3,data4)
            test_img=test_loard(taget1,taget2)

            gan_out=self.generator.predict(trains_count)
            real=np.ones((2,)+(16,16,1))
            fack=np.zeros((2,)+(16,16,1))

            real_loss=self.cnn_model.train_on_batch(test_img,real)
            fack_loss=self.cnn_model.train_on_batch(gan_out,fack)
            cnn_loss=np.add(fack_loss,real_loss)*0.5

            vgg_out=self.vgg19(test_img)
            g_loss=self.end_model.train_on_batch(trains_count,[real,vgg_out])
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, feature loss: %05f] " \
                  % (epoch, epochs,
                     cnn_loss[0], 100 * cnn_loss[1],
                     g_loss[1],
                     g_loss[2]
                     ))
            if epoch == epochs :
                self.generator.save_weights("./weights_gan_50000.h5")
                self.cnn_model.save_weights("./weights_cnn_50000.h5")
    def see(self):
        self.generator.load_weights("./weights_gan_12000.h5")
        data1, data2, data3, data4, taget1, taget2 = train_load()
        print(data1, data2, data3, data4, taget1, taget2 )
        end(data1, 12)
        end(data2, 14)
        end(data3, 15)
        end(data4, 17)
        end(taget1, 13)
        end(taget2, 16)
        data1 = loard(data1)
        data2 = loard(data2)
        data3 = loard(data3)
        data4 = loard(data4)
        conut=cont(data1,data2,data3,data4)
        imgs=self.generator.predict(conut)
        img=np.array(imgs[0])
        img1=np.array(imgs[1])
        img1=(img1+1)*127.5
        img=(img+1)*127.5
            #img=tf.image.resize(img[0],size=[512,512])
            #imgs_hr = tf.keras.preprocessing.image.array_to_img(img[0])
            #pred_img = tf.image.convert_image_dtype(img, tf.uint8)
        pred_data = tf.image.encode_jpeg(img)
        pred_data1 = tf.image.encode_jpeg(img1)
        path='./png/%d_00.jpg'
        path1 = './png/%d_11.jpg'
        tf.io.write_file(path, pred_data)
        tf.io.write_file(path1, pred_data1)
            #if b==10396:
                #break
            #plt.imshow(img)
            #plt.imsave(path,img,dpi=500)
    def use(self,path):
        self.generator.load_weights("./weights_gan_12000.h5")
        strs1,strs2,strs3=path+'/[0-9].jpg',path+'/[0-9][0-9].jpg',path+'/[0-9][0-9][0-9].jpg'
        path=glob.glob(strs1)
        path+=glob.glob(strs2)
        path+=glob.glob(strs3)
        a,b=0,1
        c=2
        for i in range(1500):
            print('第%d次'%i)
            img1=path[a]
            img2=path[b]
            gt=self.generator.predict(test_cont(img1,img2))
            img = np.array(gt[0])
            img = (img + 1) * 127.5
            pred_data = tf.image.encode_jpeg(img)
            paths='./png/%d.jpg'%c
            tf.io.write_file(paths, pred_data)
            a=b
            b+=1
            c+=2






gan=TQGAN()
#gan.train(60000)
#gan.see()
#gan.generator.save_weights("./weights_gan_12000.h5")
gan.use('./img5')


















