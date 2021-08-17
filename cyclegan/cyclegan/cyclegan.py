from tensorflow.keras.layers import *
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.models import Model
from data import load_data
import tensorflow.keras.backend as B
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from resnet import resnet
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


class cyclegan():
    def __init__(self):
        self.optimizer=tf.keras.optimizers.Adam(0.0002,0.5)
        #判别网络的特征图大小
        self.cnn_map=128//2**4

        #构建损失权重，风格损失，自我合成损失
        self.cycle_weits=5
        self.img_self=2.5


        self.img_shape=(128,128,3)
        #生成网络
        self.A_B=self.generator()
        self.B_A=self.generator()

       #判别网络
        self.see_A=self.cnn_model()
        self.see_B=self.cnn_model()
        self.see_A.compile(loss='mse',metrics=['acc'],optimizer=self.optimizer)
        self.see_B.compile(loss='mse',metrics=['acc'],optimizer=self.optimizer)

        #结合网络
        img_A=Input(shape=self.img_shape)
        img_B=Input(shape=self.img_shape)
        #生成假的A,B图片
        g_B=self.A_B(img_A)
        g_A=self.B_A(img_B)

        #再反转生成
        fack_A=self.B_A(g_B)
        fack_B=self.A_B(g_A)

        #设置判别网络不训练
        self.see_B.trainable=False
        self.see_A.trainable=False

        #构造风格图片经过生成网络A_B不变和真实图片经过B——A不变
        real_B=self.A_B(g_B)
        real_A=self.B_A(g_A)

        # 判断生成的假风格图片和真实图片
        valid_B = self.see_B(g_B)
        valid_A = self.see_A(g_A)

        self.end_model=Model(inputs=[img_A,img_B],
                             outputs=[valid_A,valid_B,
                                      fack_A,fack_B,
                                      real_A,real_B])
        self.end_model.compile(loss=['mse','mse',
                                     'mae','mae',
                                     'mae','mae'],
                               loss_weights=[0.5,0.5,
                                             self.cycle_weits,self.cycle_weits,
                                             self.img_self,self.img_self],
                               optimizer=self.optimizer)

    def generator(self):
        model=resnet()
        return model

    def cnn_model(self):
        def conv2d(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if normalization:
                d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img = Input(shape=self.img_shape)
        # 64,64,64
        d1 = conv2d(img, 64, normalization=False)
        # 32,32,128
        d2 = conv2d(d1, 128)
        # 16,16,256
        d3 = conv2d(d2, 256)
        # 8,8,512
        d4 = conv2d(d3, 512)
        # 对每个像素点判断是否有效
        # 64
        # 8,8,1
        validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d4)

        return Model(img, validity)
    def scheduler(self,models,epoch):
        # 每隔100个epoch，学习率减小为原来的1/2
        if epoch%5000==0 and epoch != 0:
            for model in models:
                lr = B.get_value(model.optimizer.lr)
                B.set_value(model.optimizer.lr, lr * 0.5)
                print("lr changed to {}".format(lr * 0.5))

    def tarin(self,batch,epochs):
        ons=np.ones((batch,)+(self.cnn_map,self.cnn_map,1))
        zeros=np.zeros((batch,)+(self.cnn_map,self.cnn_map,1))
        for epoch in range(epochs):
            self.scheduler(models=[self.end_model,self.see_A,self.see_B],epoch=epoch)
            img_tagets,img_styels=load_data()
            g_loss=self.end_model.train_on_batch([img_tagets,img_styels],
                                          [ons,ons,
                                           img_tagets,img_styels,
                                           img_tagets,img_styels])
            gan_B=self.A_B.predict(img_tagets)
            gan_A=self.B_A.predict(img_styels)

            realA_loss=self.see_A.train_on_batch(img_tagets,ons)
            fackA_loss=self.see_A.train_on_batch(gan_A,zeros)
            cnn_A_loss=np.add(realA_loss,fackA_loss)*0.5

            realB_loss=self.see_B.train_on_batch(img_styels,ons)
            fackB_loss=self.see_B.train_on_batch(gan_B,zeros)
            cnn_B_loss=np.add(realB_loss,fackB_loss)*0.5

            cnn_loss=np.add(cnn_A_loss,cnn_B_loss)*0.5

            print('[epoch:%d]--[判别损失:%05f]--[准确率:%.4f%%]--[风格转换损失:%05f]--[自我转换损失:%05f]'
                  %(epoch,cnn_loss[0],cnn_loss[1]*100,np.add(g_loss[2],g_loss[3])*0.5,np.add(g_loss[4],g_loss[5])*0.5))

            if epoch==epochs-1:
                self.B_A.save_weights('./wei/B_A%d.h5'%epochs)
                self.A_B.save_weights('./wei/A_B%d.h5'%epochs)
                self.see_A.save_weights('./wei/see_A%d.h5'%epochs)
                self.see_B.save_weights('./wei/see_B%d.h5'%epochs)
    def test(self,number):
        self.A_B.load_weights('./wei/A_B57000.h5')
        self.B_A.load_weights('./wei/B_A57000.h5')
        for i in range(number):
           img_taget,img_styel=load_data()
           img_styels=self.A_B.predict(img_taget)
           img_tagets=self.B_A(img_styels)
           # pred_img = tf.image.convert_image_dtype(img_styels[0], tf.uint8)
           # pred_data = tf.image.encode_png(pred_img)
           # tf.io.write_file('./1.png', pred_data)
           plt.imshow(tf.image.resize(img_taget[0],[512,512]))
           plt.title('img_taget')
           plt.show()
           plt.imshow(tf.image.resize(img_styel[0],[512,512]))
           plt.title('img_styel')
           plt.show()
           plt.imshow(tf.image.resize(img_styels[0],[512,512]))
           plt.title('GAN_styels')
           plt.show()
           plt.imshow(tf.image.resize(img_tagets[0],[512,512]))
           plt.title('GAN_tagets')
           plt.show()
gan=cyclegan()
#gan.tarin(batch=1,epochs=57000)
gan.test(3)













