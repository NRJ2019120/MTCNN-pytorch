import numpy as np
import iou_utils
from PIL import Image
import os

IMAGE_PATH = r'/home/tensorflow01/oneday/img_celeba'
LABEL_FILE_PATH = r'/home/tensorflow01/oneday/celeba/little_bbox_celeba.txt'
LANDMARTS_FILE_PATH = r"/home/tensorflow01/oneday/celeba/Anno/list_landmarks_celeba.txt"
sample_path = r'/media/tensorflow01/myfile/landmarks_img_celeba'
def mkdir(size):  #创建文件夹
    rootpath = os.path.join(sample_path, str(size))
    if not os.path.exists(rootpath):
        os.mkdir(rootpath)

    p_img_path = os.path.join(rootpath, "positive")
    if not os.path.exists(p_img_path):
        os.mkdir(p_img_path)

    n_img_path = os.path.join(rootpath, "negative")
    if not os.path.exists(n_img_path):
        os.mkdir(n_img_path)

    part_img_path = os.path.join(rootpath, "part")
    if not os.path.exists(part_img_path):
        os.mkdir(part_img_path)

    return rootpath, p_img_path, n_img_path, part_img_path

def gen_sample(size):

    root_path, p_img_path, n_img_path, part_img_path = mkdir(size)  # 创建图片保存文件
    p_lebal_file = open(root_path + "/positive.txt", "w")  #创建标签文件
    n_lebal_file = open(root_path + "/negative.txt", "w")
    part_lebal_file = open(root_path + "/part.txt", "w")

    count = 0
    for (i, lebal), (epoch, landmarks) in zip(enumerate(open(LABEL_FILE_PATH).readlines()),enumerate(open(LANDMARTS_FILE_PATH).readlines())):
        #enumerate :列举,数
        # enumerate(iterable[, start]) -> iterator for index, value of iterable 计数默认从0开始
        if i > 1 and i< 202599+2:                              #i>1    图片标签从第三行开始.试验7张图片
            # strs = list(filter(bool,line.split(" ")))
            strs = lebal.strip().split()##与上一行同效果,返回的是列表
            landmarks = landmarks.strip().split()
            if i % 1000 == 0:
                print(strs)
                print(landmarks)

            filename = strs[0].strip() #Return a copy of the string S with leading and trailing ,whitespace removed.
            x1 = int(strs[1].strip())
            y1 = int(strs[2].strip())
            w = int(strs[3].strip())
            h = int(strs[4].strip())
            lefteye_x = int(landmarks[1].strip())
            lefteye_y = int(landmarks[2].strip())
            righteye_x = int(landmarks[3].strip())
            righteye_y = int(landmarks[4].strip())
            nose_x = int(landmarks[5].strip())
            nose_y = int(landmarks[6].strip())
            leftmouth_x = int(landmarks[7].strip())
            leftmouth_y = int(landmarks[8].strip())
            rightmouth_x = int(landmarks[9].strip())
            rightmouth_y = int(landmarks[10].strip())

            if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:    #剔除噪声
                continue
            x2 = x1 + w
            y2 = y1 + h

            cx = x1 + w/2
            cy = y1 + h/2

            for _ in range(2):               #主要生成正样本
                dx = np.random.uniform(-0.14,0.14)
                # print(dx)
                dy = np.random.uniform(-0.14,0.14)
                d_len = np.random.uniform(-0.03,0.1)  #有问题，框缩的太小

                # _cx = cx * (1 + dx)                #中心点百分比浮动,
                # _cy = cy * (1 + dy)
                _cx = cx + w*dx                       #生成正样本，部分样本效果更好，可以体会,但是正样本生成慢
                _cy = cy + h*dy

                side_len = np.maximum(w, h)*(1 + d_len) #框大小百分比浮动  0.9~1.1

                _x1 = _cx - side_len/2                #浮动框坐标计算
                _y1 = _cy - side_len/2
                _x2 = _x1 + side_len
                _y2 = _y1 + side_len

                _lefteye_x = (lefteye_x - _x1 )/ side_len   #相对坐标 ，关键点不用偏移，直接输出预测关键点坐标
                _lefteye_y = (lefteye_y - _y1)/side_len
                _righteye_x = (righteye_x - _x1)/side_len
                _righteye_y = (righteye_y - _y1)/side_len
                _nose_x = (nose_x - _x1)/side_len
                _nose_y = (nose_y - _y1)/side_len
                _leftmouth_x = (leftmouth_x - _x1)/side_len
                _leftmouth_y = (leftmouth_y - _y1)/side_len
                _rightmouth_x = (rightmouth_x - _x1)/side_len
                _rightmouth_y = (rightmouth_y - _y1)/side_len

                box = np.array([_x1,_y1,_x2,_y2,0])  #样本浮动正方形框

                boxes = np.array([[x1,y1,x2,y2,0]])  #boxes = [[   ]]真实标签

                # _box = iou_utils.rect2squar(np.array([box]))[0]  # ??? 见test   样本浮动框转成正方形框

                # 计算偏移值                           #???存在问题,偏移百分比   #运行时警告：在双标量中遇到的除以零?

                offset_x1 = (x1 - _x1) / side_len     #运行时警告：在双标量中遇到的除以零
                offset_y1 = (y1 - _y1) / side_len      #即是正方形框边长
                offset_x2 = (x2 - _x2) / side_len
                offset_y2 = (y2 - _y2) / side_len


                im = Image.open(os.path.join(IMAGE_PATH,filename))#读取标签指定图片
                im = im.crop(box[0:4])            #图片截取,修剪图片
                im = im.resize((size,size))        #重新定尺寸 需要更好的理解 ,生成样本

                iou = iou_utils.iou(box,boxes)
                if iou[0] > 0.65: #正样本,负样本置信度标签1, 0.  原论文0.65
                    count += 1
                    im.save("{0}/{1}.jpg".format(p_img_path,count))
                    p_lebal_file.write(
                        "{0}.jpg  1  {1}  {2}  {3}  {4}  {5}  {6}  {7}  {8}  {9}  {10}  {11}  {12}  {13}  {14}\n".format(
                            count, offset_x1, offset_y1, offset_x2, offset_y2,_lefteye_x,_lefteye_y,_righteye_x,
                            _righteye_y,_nose_x,_nose_y,_leftmouth_x,_leftmouth_y,_rightmouth_x,_rightmouth_y))
                # elif iou[0] > 0.4:             #部分 样本
                #     count += 1
                #     im.save("{0}/{1}.jpg".format(part_img_path, count))
                #     part_lebal_file.write(
                #         "{0}.jpg  2  {1}  {2}  {3}  {4}  {5}  {6}  {7}  {8}  {9}  {10}  {11}  {12}  {13}  {14}\n".format(
                #             count, offset_x1, offset_y1, offset_x2, offset_y2,_lefteye_x,_lefteye_y,_righteye_x,
                #             _righteye_y,_nose_x,_nose_y,_leftmouth_x,_leftmouth_y,_rightmouth_x,_rightmouth_y))

            for _ in range(3):               #主要生成部分样本
                dx = np.random.uniform(-0.25,0.25)
                dy = np.random.uniform(-0.25,0.25)
                d_len = np.random.uniform(-0.1,0.1)

                _cx = cx * (1 + dx)                #中心点百分比浮动,
                _cy = cy * (1 + dy)
                # _cx = cx + w*dx                       #生成正样本，部分样本效果更好，可以体会,但是正样本生成慢
                # _cy = cy + h*dy

                side_len = np.maximum(w, h)*(1 + d_len) #框大小百分比浮动

                _x1 = _cx - side_len/2                #浮动框坐标计算
                _y1 = _cy - side_len/2
                _x2 = _x1 + side_len
                _y2 = _y1 + side_len

                _lefteye_x = (lefteye_x - _x1 )/ side_len   #相对坐标 ，关键点不用偏移，直接输出预测关键点坐标
                _lefteye_y = (lefteye_y - _y1)/side_len
                _righteye_x = (righteye_x - _x1)/side_len
                _righteye_y = (righteye_y - _y1)/side_len
                _nose_x = (nose_x - _x1)/side_len
                _nose_y = (nose_y - _y1)/side_len
                _leftmouth_x = (leftmouth_x - _x1)/side_len
                _leftmouth_y = (leftmouth_y - _y1)/side_len
                _rightmouth_x = (rightmouth_x - _x1)/side_len
                _rightmouth_y = (rightmouth_y - _y1)/side_len

                box = np.array([_x1,_y1,_x2,_y2,0])  #样本浮动正方形框

                boxes = np.array([[x1,y1,x2,y2,0]])  #boxes = [[   ]]真实标签

                # _box = iou_utils.rect2squar(np.array([box]))[0]  # ??? 见test   样本浮动框转成正方形框

                # 计算偏移值                           #???存在问题,偏移百分比   #运行时警告：在双标量中遇到的除以零?

                offset_x1 = (x1 - _x1) / side_len     #运行时警告：在双标量中遇到的除以零
                offset_y1 = (y1 - _y1) / side_len      #即是正方形框边长
                offset_x2 = (x2 - _x2) / side_len
                offset_y2 = (y2 - _y2) / side_len


                im = Image.open(os.path.join(IMAGE_PATH,filename))#读取标签指定图片
                im = im.crop(box[0:4])            #图片截取,修剪图片
                im = im.resize((size,size))        #重新定尺寸 需要更好的理解 ,生成样本

                iou = iou_utils.iou(box,boxes)
                if iou[0] > 0.65: #正样本,负样本置信度标签1, 0.  原论文0.65
                    count += 1
                    im.save("{0}/{1}.jpg".format(p_img_path,count))
                    p_lebal_file.write(
                        "{0}.jpg  1  {1}  {2}  {3}  {4}  {5}  {6}  {7}  {8}  {9}  {10}  {11}  {12}  {13}  {14}\n".format(
                            count, offset_x1, offset_y1, offset_x2, offset_y2,_lefteye_x,_lefteye_y,_righteye_x,
                            _righteye_y,_nose_x,_nose_y,_leftmouth_x,_leftmouth_y,_rightmouth_x,_rightmouth_y))  #和老师不一样
                elif iou[0] > 0.4:             #部分 样本
                    count += 1
                    im.save("{0}/{1}.jpg".format(part_img_path, count))
                    part_lebal_file.write(
                        "{0}.jpg  2  {1}  {2}  {3}  {4}  {5}  {6}  {7}  {8}  {9}  {10}  {11}  {12}  {13}  {14}\n".format(
                            count, offset_x1, offset_y1, offset_x2, offset_y2,_lefteye_x,_lefteye_y,_righteye_x,
                            _righteye_y,_nose_x,_nose_y,_leftmouth_x,_leftmouth_y,_rightmouth_x,_rightmouth_y))

                elif iou[0] < 0.23:#负样本
                    count += 1
                    im.save("{0}/{1}.jpg".format(n_img_path, count))
                    n_lebal_file.write(
                        "{0}.jpg  0  {1}  {2}  {3}  {4}  {5}  {6}  {7}  {8}  {9}  {10}  {11}  {12}  {13}  {14}\n".format(
                            count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)) #没有偏移量

            for _ in range(9):  # 再多造一些负样本
                im = Image.open(os.path.join(IMAGE_PATH, filename))
                img_wight,img_height = im.size              #size 是属性
                d_len = np.random.uniform(-0.2,0.2)
                side_len = np.maximum(w, h)*(1 + d_len)  # 框大小百分比浮动
                bound_w = max(1,img_wight - side_len)
                bound_h = max(1,img_height - side_len)
                _x1 = np.random.randint(0,bound_w)
                _y1 = np.random.randint(0,bound_h)
                _x2 = _x1 + side_len                      #直接生成正方形
                _y2 = _y1 + side_len

                box = np.array([_x1, _y1, _x2, _y2, 0])  #直接生成正方形浮动框
                boxes = np.array([[x1, y1, x2, y2, 0]])

                iou = iou_utils.iou(box, boxes)
                if iou[0] < 0.22:            # 负样本
                    count += 1
                    im = im.crop(box[0:4])  # 图片截取,修剪图片
                    im = im.resize((size, size))  # 生成负样本
                    im.save("{0}/{1}.jpg".format(n_img_path, count))
                    n_lebal_file.write(
                        "{0}.jpg  0  {1}  {2}  {3}  {4}  {5}  {6}  {7}  {8}  {9}  {10}  {11}  {12}  {13}  {14}\n".format(count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            # # break
    p_lebal_file.close()
    n_lebal_file.close()
    part_lebal_file.close()
if __name__ == '__main__':

    gen_sample(12)
    gen_sample(24)
    gen_sample(48)






