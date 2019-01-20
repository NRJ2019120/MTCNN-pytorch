import os
from PIL import Image
import numpy as np

LABEL_FILE_PATH = r'/home/tensorflow01/oneday/celeba/Anno/list_bbox_celeba.txt'
IMAGE_PATH = r'/home/tensorflow01/oneday/celeba/img_celeba'
img_celeba_path = r"/home/tensorflow01/oneday/celeba"

def crop_celeba():
    little_celeba_path = os.path.join(img_celeba_path,"little_celeba")
    if not os.path.exists(little_celeba_path):
        os.makedirs(little_celeba_path)
    list_file = open(img_celeba_path +"/little_list_celeba","w")
    for i, line in enumerate(open(LABEL_FILE_PATH).readlines()):
        """index = filename ,x ,y ,w,h"""
        if i ==0:
            list_file.write("32317\n")
        if i ==1:
            list_file.write("image_id x1 y1 width height\n")
        if i>1 and i < 32319:

            strs = line.strip().split()
            if i % 1000 == 0:
                print(strs)
            filename = strs[0]

            cx =int(strs[1]) + int(strs[3])/2  #人脸框中心点坐标
            cy =int(strs[2]) + int(strs[4])/2

            w = int(strs[3])
            h = int(strs[4])

            _w = int(strs[3])*1.25          #扩大1.25
            _h = int(strs[4])*1.25

            side_len = int(np.maximum(_w,_h))

            x1 = int(cx - side_len/2)
            y1 = int(cy - side_len/2)
            x2 = x1 + side_len
            y2 = y1 + side_len

            crop_box = [x1, y1, x2, y2] #裁剪区域

            _x = int(strs[1]) - x1  #裁剪后相对坐标
            _y = int(strs[2]) - y1

            list_file.write("{0}  {1}  {2}  {3}  {4} \n".format(filename,_x,_y,w,h))

            image = Image.open(os.path.join(IMAGE_PATH, filename))

            image = image.crop(crop_box)
            image.save("{0}/{1}".format(little_celeba_path,filename))

    list_file.close()

if __name__ == '__main__':
    little_celeba()





