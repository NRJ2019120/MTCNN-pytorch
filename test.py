import numpy as np
from PIL import Image
import torch
import torchvision
# x = np.array([1,2,3,4])
# y = np.array([5,6,7,8])
# print(np.concatenate([x,y],axis=0))# concatenate vt.	把 （一系列事件、事情等）联系起来;adj.连接的，联系在一起的;
#
# print(np.stack([x,y],axis=1))  #比较区别
# stack#	垛，干草堆; （一排） 烟囱; 层积; 整个的藏书架排列;
# vt.	堆成堆，垛; 堆起来或覆盖住; 洗牌作弊; 秘密事先运作;


# from PIL import Image,ImageDraw
# im = Image.open(r"D:\cebela\img_celeba\000004.jpg")
# imDraw = ImageDraw.Draw(im)
# imDraw.rectangle((622,257,622+564,257+781),outline='red')
# im.show()
#
# lines = "000003.jpg   216  59  91 126"
# strs = lines.strip().split(" ")
# print(strs)
# print(bool(""))
# strs = list(filter(bool, strs))
# print(strs)
# print(lines.strip().split())
"""广播规则"""
# a = np.array([[5,5,15,15,1]])
# x1 = a[:,0]
# y1 = a[:,1]
# x2 = a[:,2]
# y2 = a[:,3]
# b = np.stack([x1, y1, x2, y2,a[:,4]], axis=1)
# print(b[0])
# print(b)          #广播规则
# c = np.array([0,0,10,10,1])

# print(np.maximum(0,c))#注意np.maximum()与np.max()的区别_
# print(np.max(0,c))    #报错

# def iou(box,boxes,mode="UNION"):        #iou重叠度计算
#
#     top_x = np.maximum(box[0],boxes[:,0])#涉及广播规则,切片
#     top_y = np.maximum(box[1],boxes[:,1])
#     bottom_x = np.minimum(box[2],boxes[:,2])
#     bottom_y = np.minimum(box[3],boxes[:,3])
#
#     w = np.maximum(0, (bottom_x - top_x))
#     h = np.maximum(0, (bottom_y - top_y))  # 注意np.maximum()与np.max()的区别_
#
#     j_area = w * h
#     box_area = (box[2]-box[0])*(box[3]-box[1])           #框面积
#     boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])#框面积 张量
#
#     if mode == "UNION":
#         fm_area = box_area + boxes_area - j_area            #两框并集面积张量
#         return j_area/fm_area
# print(iou(c,a)[0])                                            #iou[0]
# print(iou(c,a))                                                #是一个列表
#
# im = Image.open("D:\cebela\img_align_celeba_007/000001.jpg","r")
# img_wight, img_hight = im.size
# print(im.size)
#
# # 000001.jpg    95  71 226 313 x1,y1 ,wight,hight,置信度
# boxes = np.array([[95,71,226,313,0]])
# print(boxes)
# print(boxes[:,0])
# def rect2squar(boxes):            #boxes为一图多个人脸多个框情况,boxes为二维张量,把框变成正方形框
#     w = boxes[:, 2] - boxes[:, 0]
#     print(w)
#     h = boxes[:, 3] - boxes[:, 1]
#     print(h)
#     side_len = np.maximum(w, h)
#     cx = boxes[:, 0] + w / 2      #中心点x 坐标
#     cy = boxes[:, 1] + h / 2      #中心点y 坐标
#
#     x1 = cx - side_len / 2
#     y1 = cy - side_len / 2
#     x2 = cx + side_len / 2
#     y2 = cy + side_len / 2
#     return np.stack([x1, y1, x2, y2,boxes[:,4]], axis=1) #注意此时是轴为1上的拼接
    # np.concatenate([x1,y1,x2,y2],axis=0)的区别
# print("boxes",rect2squar(boxes))
"""list.extend"""
# list= []
# list1 = [1,2,3]
# list2 = [4,5,6]
# list.extend(list1)
# list.extend(list2)
# print(list)
"""IMAGE.resize"""
# path = r"D:\cebela\img_celeba\000001.jpg"
# im = Image.open(path)
# im.show()
#
# im = im.crop((95 , 71, 226+95, 313+71))
# im.show()
# im=im.resize((1200,1000))
# im.show()
# img_data = np.array(im)
# print(img_data)
# #掩码
# a=torch.Tensor([1,2,3,4])
# MASK = torch.le(a,3)
# print(a[MASK])
# import os
# os.makedirs(r"D:\cebela\img_align_celeba_007\params")

# a = torch.Tensor(int(1.0))
# print(a)
# x = [[[[1,2,3,4]]]]
# print(x[0][0][0])
# img = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
# print(img.shape)
# print(img.shape[0])
# a = torch.Tensor([1,3])
# print(a)
# print(a.shape)
# print(a.size())
# b= a.unsqueeze(0)
# print(b)
# print(b.size())
# x= torch.rand(1,3)
# y= torch.rand(1,3)
# print(x)
# print(y)
# c=torch.stack((x,y),0)
# print(c)
# a = torch.Tensor([[4.],
#         [2.8],
#         [2.5],
#         [1.54],
#         [1.6127],
#         [1.67]])
# print(a.size())
# b = torch.gt(a, 2.5)
# print(b)
# c = torch.nonzero(torch.gt(a, 2.5))
# d = torch.nonzero(b)
# print(c)
# print(d)
# print(a)
# print(a.size())
# print(a[0])
# print(a[0].cpu().data)
# print(a[0].size())
# print(a[0][0])
# print(a[0][0].size())
# print(a[0][0].cpu().data) #与a[0][0]没区别
# x =[[[1,2,3,4]]]
# print(x[0][0][0])
# print(x[0][0])
# print(x[0])
# a = np.array(np.random.uniform(-0.2,0.2,[3,3]))
# print(a)
# b = np.array(np.random.uniform(-0.2,0.2,[3,3]))
# a = np.maximum(2,2)
# print(a)
# str = "00001.jpg"
# print("{0}".format(str))
# img = Image.open(r"/home/tensorflow01/oneday/celeba/test_image/1.jpg")
# img = img.crop((-30,40,50,60))
# img.show()
# im = Image.open(r"/home/tensorflow01/oneday/celeba/little_celeba/032317.jpg")#读取图片
# im.show()
# print(type(im))
# 032317.jpg  15  20  114  158
# im.show()
# img = im.crop((15.8 , 20.9 , 114.5 , 158.9))
# img.show()
# imgdata2 = np.array(img)/255
# print(imgdata2)
# image_transfrom = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# img_data = image_transfrom(img)
# print(img_data)
# a = torch.Tensor([ 0.0266, -0.1543, -0.9706,  0.1502]).long()
# print(a)
# print(a.type())
# x = a[0]*2+9
# print(x)
# print(x.type())
# b = x.long()
# print(b)
# print(b.type())
# cls = torch.Tensor([0.5,0.6,0.2,0.1,2,6])
# a = np.array(cls)
# print(a)
# idxs,_ = np.where(a > 0.8)
# print(idxs)
# print(i)
# box =torch.Tensor([1,2,3,4])
# box_ = torch.Tensor([1.5,2.5,3.5,4.5])
# con = torch.Tensor([1])
# con_ = torch.Tensor([1])
# loss_fun = torch.nn.BCELoss()
# loss = loss_fun(con,con_)
# print(loss)
# x = torch.Tensor(5,3)
# x= torch.randn(6)
# print(x)
# y = x.view(-1,6)
# print(y)
# print(x.size())
# x = torch.Tensor([1.01])
# y = torch.Tensor([1])
# print(x.type())
# print(y.type())

# file = open(r'/home/tensorflow01/oneday/celeba/img_celeba/48/positive.txt',"r")
# cls = np.array([1,2,3,4,5])
# a = np.where(cls>3)
# print(cls[a])
# x = np.random.randint(0,0)
# sample_path = r'/home/tensorflow01/oneday/celeba/landmarks_img_celeba/48/positive.txt'
# file = open(sample_path,"r")
# strs =file.readline()
# print(strs)
# file = open('/media/tensorflow01/myfile/landmarks_img_celeba/12/nonlandmarks_positive.txt',"r")

#
# x = torch.linspace(1, 27, steps=27).view(9, 3)
# print(x)
# print(torch.full_like(x, 5))
# bbb = torch.where(x > 5, torch.full_like(x, 5), x)

# print(bbb)

# x1 = np.arange(5,15).reshape(10,1)
# x2 = np.arange(5,15).reshape(10,1)
# print(x1,x2)
# idx = np.equal(x1,x2)
# print(idx)
# print(x1[idx])
# mask = np.where((x1>10)&(x1<13))
# print(mask)
# print(x1[mask])
# np.where(x1[mask]<13)

x1 = torch.Tensor(np.arange(5,15).reshape(10,1))
x2 = torch.Tensor(np.arange(5,15).reshape(10,1))
mask1 = torch.gt(x1,5)
mask2 = torch.lt(x1,7)

# mask3 = torch.where(x1>5,torch.full_like(x1,1),x2=x1)
# print(mask3)
print(mask1,mask2)
# mask = torch.equal(mask1,mask2,dim=1)
# print(mask)
x1_1 = x1[mask1[:,0]]
print(x1_1)
x1_2 = x1[mask2[:,0]]
print(x1_2)
# x1_3 = x1[mask[:]]