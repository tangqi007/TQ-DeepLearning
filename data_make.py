import cv2
import os.path as op
import os
import glob
video_path='./lp.mp4'
out_path='./img'

def split_video(video_path, out_path):
    if not op.exists(out_path):
        os.mkdir(out_path)
    vc = cv2.VideoCapture(video_path)
    success, frame = vc.read()
    i=0
    while success:
        i += 1
        img_path = f'{out_path}\\{i}.jpg'
        cv2.imwrite(img_path, frame)
        if success:
            print(f'\r Split image{i}', end='')
            success, frame = vc.read()


def make_img(number,path):
    fp=open('train.txt','w')
    fs=open('taget.txt','w')
    a,b=1,2
    for i in range(number):
        img_path=path+'/%d.jpg\n'%a
        taget_path=path+'/%d.jpg\n'%b
        fs.write(taget_path)
        fp.write(img_path)
        a=b+1
        b=a+1
    fp.close()
    fs.close()
#d=glob.glob('D:/迅雷下载/240fps/original_high_fps_videos/*.mp4')
split_video('D:/迅雷下载/240fps/original_high_fps_videos/IMG_0054a.mov','./img9')
