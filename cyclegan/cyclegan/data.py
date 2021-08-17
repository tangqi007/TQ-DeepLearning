import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import pylab
import random


def load_data():
        path1 = glob.glob('./dataset/monet2photo/trainB/*.jpg')
        path2 = glob.glob('./dataset/monet2photo/trainA/[0-9][0-9][0-9][0-9][0-9].jpg')

        img_t = random.choice(path1)
        img_s = random.choice(path2)

        img_tagets = []
        img_styels = []

        img_taget = imread(img_t)
        img_styel= imread(img_s)

        h, w = (128,128)

        img_taget =np.array(img_taget.resize((h,w)))
        img_styel =np.array(img_styel.resize((h,w)))


        img_tagets.append(img_taget)
        img_styels.append(img_styel)

        img_tagetss = np.array(img_tagets) /127.5-1
        img_styelss= np.array(img_styels) / 127.5-1

        return img_tagetss, img_styelss

def imread(path):
    return Image.open(path)
# d=DataLoader()
# x,y=load_data()
# plt.imshow(x[0])
# pylab.show()
# plt.imshow(y[0])
# pylab.show()

