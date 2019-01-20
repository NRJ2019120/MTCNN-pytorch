from PIL import Image
import os
from PIL import ImageDraw
label_file = r"/home/tensorflow01/oneday/celeba/list_bbox_celeba.txt"
little_label_file = r"/home/tensorflow01/oneday/celeba/little_bbox_celeba.txt"
img_landmark_file = r"/home/tensorflow01/oneday/celeba/Anno/list_landmarks_celeba.txt"
img_label_file = r"/home/tensorflow01/oneday/celeba/Anno/list_bbox_celeba.txt"
img_path = r":/media/tensorflow01/myfile/img_celeba"


for (i,lebal),(epoch,landmark) in zip(enumerate(open(little_label_file).readlines()),enumerate(open(img_landmark_file).readlines())):
    if i > 1 and i <3:
        print(i,lebal,epoch,landmark)
        # break
# for i,line in enumerate(open(img_label_file).readlines()):

    if i >105 and i <115:
        print(lebal)
        strs = lebal.split()
        landmark = landmark.split()
        print(strs)
        print(landmark)
        filename = strs[0]
        print(filename)
        x1 = int(strs[1])
        y1 = int(strs[2])
        w = int(strs[3])
        h = int(strs[4])

        lefteye_x = int(landmark[1])
        lefteye_y = int(landmark[2])
        righteye_x = int(landmark[3])
        righteye_y = int(landmark[4])
        nose_x = int(landmark[5])
        nose_y = int(landmark[6])
        leftmouth_x = int(landmark[7])
        leftmouth_y = int(landmark[8])
        rightmouth_x = int(landmark[9])
        rightmouth_y = int(landmark[10])

        im = Image.open(os.path.join(img_path,filename))#读取图片
        print(im.size)
        imDraw = ImageDraw.Draw(im)
        imDraw.rectangle((x1,y1,x1+w,y1+h),outline="red")
        # imDraw.rectangle((x1,y1,x1+w,y1+h),outline="green")
        imDraw.polygon([(lefteye_x,lefteye_y),(righteye_x,righteye_y),(nose_x,nose_y),
                         (leftmouth_x,leftmouth_y),(rightmouth_x,rightmouth_y)],outline="black")
        # imDraw.rectangle((95,71,95+226,71+313),outline="red")
        im.show()


