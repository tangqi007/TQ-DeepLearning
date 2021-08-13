import cv2
import os
im_dir = './png'  # 图片存储路径
video_dir = './new.avi' #合成后的视频名称, 只能合成avi格式视频
data=[]
for i in range(1,1192):
    str='./png/%d.jpg'%i
    data.append(str)
#imglist = sorted(os.listdir(im_dir)) #将排序后的路径返回到imglist列表中
#img = cv2.imread(os.path.join(im_dir,data[0])) #合并目录与文件名生成图片文件的路径,随便选一张图片路径来获取图像大小
#H, W, D = img.shape #获取视频高\宽\深度
#print('height:' + str(H)+'--'+'width:'+str(W)+'--'+'depth:'+str(D))
fps =102 #帧率一般选择20-30
img_size = (256,256) #图片尺寸宽x高,必须是原图片的size,否则合成失败
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
for image in data:
    img_name = os.path.join(im_dir, image)
    frame = cv2.imread(image)
    videoWriter.write(frame)
    print('合成==>'+image)
videoWriter.release()
print('finish!')