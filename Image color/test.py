import tensorflow as tf
import model
from config import *
import os
import numpy as np
test_file = './img/15.jpg'
save_file = './test/15_p.png'

print('正在加载图片')

img = tf.io.read_file(test_file)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, [net_input_size, net_input_size])
img = tf.cast(img, tf.float32)  # 归一化
img = (img / 127.5) - 1
img = tf.image.rgb_to_yuv(img)

gray, uv = tf.split(img, [1, 2], axis=2)

print('正在加载模型')

gen = model.Generator(input_shape=gray.shape)
gen.load_weights(GenWeightPath)

print('正在预测...')

pred = gen(tf.expand_dims(gray, axis=0))[0]
pred_img = np.concatenate([gray,pred],axis=-1)
pred_img = tf.image.yuv_to_rgb(pred_img)
#pred_img = tf.clip_by_value(pred_img, 0., 1.)
pred_img = tf.image.convert_image_dtype(pred_img, tf.uint8)
pred_data = tf.image.encode_png(pred_img)
tf.io.write_file(save_file, pred_data)