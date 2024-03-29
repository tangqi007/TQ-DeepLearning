import tensorflow as tf
from config import *
import glob


def trans_func(file):
    data = tf.io.read_file(file)
    img = tf.image.decode_png(data, channels=3) # 图片解码

    #img = tf.cast(img, tf.float32)
    # 将图片的shape转为float，因为tf中大部分运算要求运算数之间的类型相同

    #min_size = tf.math.minimum(x=img_size[0], y=img_size[1]) # 取出较小的一边
    #crop_size = tf.random.uniform([], min_size * 0.15, min_size)
    # 截取的尺寸，在min_size * 0.15, min_size之间波动，以获取到不同缩放的样本。
    #img = tf.image.convert_image_dtype(img, tf.float32) # 归一化
    #img = tf.image.random_crop(img, [crop_size, crop_size, 3]) # 随机裁剪
    img = tf.image.resize(img, [net_input_size, net_input_size]) # 缩放到固定大小
    img = tf.cast(img, tf.float32)  # 归一化
    img=(img/127.5)-1
    #img = tf.image.random_flip_left_right(img) # 随机左右翻转
    img = tf.image.rgb_to_yuv(img) # 转为yuv格式
    return img


def GetDataset(data_path, Epochs, BatchSize):
    files = glob.glob(data_path)

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(trans_func)
    dataset = dataset.shuffle(len(files))
    dataset = dataset.batch(BatchSize)
    dataset = dataset.repeat(Epochs)

    return dataset