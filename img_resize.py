import numpy as np
from PIL import Image
import glob
'''主要是把原图全变成256 256 的，方便与生成的拼接'''
def img_raed(path):
    return Image.open(path)

img_h=256
img_w=256
def resize(path):
    img=img_raed(path)
    img_data = np.array(img.resize((img_h, img_w)),dtype=np.uint8)
    return img_data
def to_img(arry,number):
    img=Image.fromarray(arry)
    img.save('./png/%d.jpg'%number)
def end(path,number):
    arry=resize(path)
    to_img(arry,number)
path='./img5'
strs1,strs2,strs3=path+'/[0-9].jpg',path+'/[0-9][0-9].jpg',path+'/[0-9][0-9][0-9].jpg'
path=glob.glob(strs1)
path+=glob.glob(strs2)
path+=glob.glob(strs3)
b=1
for i in range(len(path)):
    end(path[i],b)
    b+=2