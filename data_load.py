import numpy as np
from PIL import Image
import random

def img_raed(path):
    return Image.open(path)
#加载处理数据集
#data=open('./tarin.txt')
#test=open('./taget.txt')
#test_data=test.readlines()
#txt=data.readlines()
img_h=256
img_w=256
'''图片转数组'''
def loard(path):
    img=img_raed(path)
    img_data = np.array(img.resize((img_h, img_w))).astype(np.float)
    return img_data
'''取两张图像融合'''
def test_cont(img1,img2):
    img = []
    img1=loard(img1)
    img2=loard(img2)
    data = np.concatenate((img1, img2), axis=-1)
    img.append(data)
    img = np.array(img)
    img = img / 127.5 - 1
    return img
'''取四张图融合'''
def cont(img1,img2,img3,img4):
    img=[]
    data=np.concatenate((img1,img2),axis=-1)
    data1=np.concatenate((img3,img4),axis=-1)
    img.append(data)
    img.append(data1)
    img=np.array(img)
    img=img/127.5-1
    return  img
'''取两张目标图片'''
def test_loard(path,path1):
    test=[]
    img=img_raed(path)
    img1=img_raed(path1)
    img_data = np.array(img.resize((img_h, img_w))).astype(np.float)
    img_data1 = np.array(img1.resize((img_h, img_w))).astype(np.float)
    test.append(img_data)
    test.append(img_data1)
    test=np.array(test)
    test=test/127.5-1
    return test

def train_load():
    path=['./img/','./img1/','./img2/','./img3/','./img4/','./img5/','./img6/','./img7/','./img8/','./img9/']
    path=random.choice(path)
    if path=='./img/':
        s=np.random.randint(2,741,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img1/':
        s = np.random.randint(2,832,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img2/':
        s = np.random.randint(2,808,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img3/':
        s = np.random.randint(2,771,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img4/':
        s = np.random.randint(2,932,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img5/':
        s = np.random.randint(2,595,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img6/':
        s = np.random.randint(2,483,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img7/':
        s = np.random.randint(2,797,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img8/':
        s = np.random.randint(2,510,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    if path == './img9/':
        s = np.random.randint(2,549,2)
        train_img1, train_img2 = path + '%d.jpg' % int(int(s[0]) - 1), path + '%d.jpg' % int(int(s[0] + 1))
        taget1 = path + '%d.jpg' % int(int(s[0]))
        train_img3, train_img4 = path + '%d.jpg' % int(int(s[1]) - 1), path + '%d.jpg' % int(int(s[1]) + 1)
        taget2 = path + '%d.jpg' % int(int(s[1]))
    return train_img1,train_img2,train_img3,train_img4,taget1,taget2


