import torch
import torchnet as nets
import torchvision
import numpy as np
import nms_utils
from PIL import Image
from PIL import ImageDraw
import os
import cv2
from detector import detector
import time

if __name__ == '__main__':

    """avi"是所有系统都会支持的视频格式"""
    video_path = r"/home/tensorflow01/oneday/celeba/程潇.mp4"
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 视频存储的编码的格式
    cap_fps = cap.get(cv2.CAP_PROP_FPS)  # 获取读取视屏的帧率
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取读取视屏的宽度
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取读取视屏的高度
    cap_total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取读取视频的总帧数
    print("视频帧率，总帧数，尺寸:W,H", cap_fps, cap_total_Frames, cap_width, cap_height)
    out = cv2.VideoWriter(r"/home/tensorflow01/oneday/celeba/程潇_test.mp4",fourcc, cap_fps,(cap_width,cap_height),isColor=True)
    # videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))
    count = 0
    start = time.time()  #单位/秒
    detector = detector()
    while cap.isOpened():
        ret, frame = cap.read()
        count +=1
        if ret==True:
            # if count % 20 ==0:   #每20帧检测一次
                print(count, frame.shape(), "==========")
                # frame = cv2.flip(frame,1) # write the flipped frame
                img = Image.fromarray(frame)
                # print(img.size)
                # print(type(img),"count=",count)
                # print("img.size",img.size)
                # print(img)
                # img.show()
                # exit()
                boxes = detector.detect(img)
                #""" boxes =[[x1, y1, x2, y2, cls]]"""
                print("out_boxes================================>", boxes)
                # exit()
                # imDraw = ImageDraw.Draw(img)
                for box in boxes:
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    lefteye_x = int(box[5])
                    lefteye_y = int(box[6])
                    righteye_x = int(box[7])
                    righteye_y = int(box[8])
                    nose_x = int(box[9])
                    nose_y = int(box[10])
                    leftmouth_x = int(box[11])
                    leftmouth_y = int(box[12])
                    rightmouth_x = int(box[13])
                    rightmouth_y = int(box[14])
                    print("cls==>", box[4])
                    # imDraw.rectangle((x1, y1, x2, y2), outline='red', width=1)
                    # imDraw.ellipse([lefteye_x, lefteye_y, lefteye_x + 5, lefteye_y + 5], fill="blue")
                    # imDraw.ellipse([righteye_x, righteye_y, righteye_x + 5, righteye_y + 5], fill="blue")
                    # imDraw.ellipse([nose_x, nose_y, nose_x + 5, nose_y + 5], fill="blue")
                    # imDraw.ellipse([leftmouth_x, leftmouth_y, leftmouth_x + 5, leftmouth_y + 5], fill="blue")
                    # imDraw.ellipse([rightmouth_x, rightmouth_y, rightmouth_x + 5, rightmouth_y + 5], fill="blue")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0),2)
                    cv2.circle(frame,(lefteye_x,lefteye_y), 2,(255,0,0),-1)
                    cv2.circle(frame,(righteye_x,righteye_y), 2,(255,0,0),-1)
                    cv2.circle(frame,(nose_x,nose_y), 2,(255,0,0),-1)
                    cv2.circle(frame,(leftmouth_x,leftmouth_y), 2,(255,0,0),-1)
                    cv2.circle(frame,(rightmouth_x,rightmouth_y), 2,(255,0,0),-1)
                out.write(frame)
                # img.show()
                cv2.imshow('frame',frame)
                if len(boxes)!= 0:     #保存图片
                    cv2.imwrite(r'/home/tensorflow01/oneday/celeba/蔡依林 -怪美的超清版_detect{0}.jpg'.format(count),frame)#保存图片
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                print('第 {} 张'.format(count), end='  ')
                print("FPS of the video is {:5.2f}".format(count / (time.time() - start)))
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

