import torch
import torchnet as nets
import torchvision
import numpy as np
import nms_utils
from PIL import Image
from PIL import ImageDraw
import os

class detector():
    def __init__(self
                 ,pnet_parampath= r"/home/tensorflow01/oneday/celeba/landmarks_params/pnet.pkl"
                 ,rnet_parampath= r"/home/tensorflow01/oneday/celeba/landmarks_params/rnet.pkl"
                 ,onet_parampath= r"/home/tensorflow01/oneday/celeba/landmarks_params/onet.pkl"
                 ,iscuda=False):

        self.iscuda = iscuda
        self.pnet = nets.P_NET(istraining=False)
        self.rnet = nets.R_NET(istraining=False)
        self.onet = nets.O_NET(istraining=False)

        if self.iscuda == True:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_parampath))  #下载参数
        self.rnet.load_state_dict(torch.load(rnet_parampath))
        self.onet.load_state_dict(torch.load(onet_parampath))

        self.pnet.eval()    #测试 param 不改变
        self.rnet.eval()    #测试 param 不改变
        self.onet.eval()    #测试 param 不改变

        self.image_transfrom = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) #转换格式
        # self.__image_transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])
    def detect(self,image):

        pnet_boxes = self.pnet_detect(image)
        if self.iscuda == True:
            pnet_boxes.cuda()
        # print("pnet_boxes==>",pnet_boxes)
        if pnet_boxes.shape[0] == 0:
            return np.array([])

        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        # print("rnet_boxes==>", rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])

        onet_boxes = self.onet_detect(image, rnet_boxes)
        # print("onet_boxes==>", onet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])

        return onet_boxes

    def pnet_detect(self,image):    #image CHW   每12*12, 判断一次 ,扫描 通过卷积完成

        boxes = []
        img = image
        # print("type(img)",type(img))
        w,h = img.size
        # print(img.size)
        # exit()
        min_side_len = min(w,h)
        # print(min_side_len)
        # print("====================")
        scale = 1.0
        xx = 0
        while min_side_len > 12:      #img 转化成cHw 三维张量
            # print('=-===============================')
            # print(xx)
            xx += 1

            img_data = self.image_transfrom(img)  #图片转成TENSOR
            # print(img_data.type())
            # print(img_data)
            # img_data = variable(img_data)
            if self.iscuda == True:
                img_data.cuda()    #把数据搬到GPU
            img_data = img_data.unsqueeze(0)  #升维度 #NCHW四维张量[10, 3, 3, 3], but got input of size [3, 318, 500] instead
            # print(img_data)
            _con,_offset = self.pnet(img_data)  #[NCHW]结构
            # print("_con==>",_con,"_offset==>",_offset)

            con = _con[0][0].cpu().data           # NCHW==>HW  con = _con[0][0].cpu().data
            # print("con==>",con)
            offset = _offset[0].cpu().data        # NCHW == >CHW c 通道分别表示 x1,y1,x2,y2
            # landmark = _landmarks[0].cpu().data
            # print("offset",offset)
            idxs = torch.nonzero(torch.gt(con,0.4)) #取置信度 >0.6的框 索引   表明pnet 训练的不够好
            # print("idxs==>",idxs.type())
            for idx in idxs:
                # print("idx==>",idx.type())#idx 为list结构【544,731】 offset为TENSOR【【【】】】CHW结构
                # print(idx[1])
                # test = idx[1] * 2 / 1
                # print("test==>", test)
                # print("con[0,359]==>",con[0,359])
                # box = self.__box(idx, offset, con[idx[0],idx[1]], scale)   #  理解???
                # print("box",box)
                # print(idx, offset, con[idx[0], idx[1]], scale)
                boxes.append(self.__box(idx, offset, con[idx[0], idx[1]], scale))
            # print("boxes==>",boxes)

            scale = scale*0.702       #图像金字塔 0.702 效果都不一样 0.9,0.8?? 思考
            _w = int(w * scale)
            _h = int(h * scale)

            img = image.resize((_w, _h))
            # print("type(img)==>",type(img),"img==>",img)
            min_side_len = min(_w, _h)
            # print("min_side_len",min_side_len)
            # break
        return nms_utils.nms(np.array(boxes), 0.6)  #N V

    def rnet_detect(self,image,pnet_boxes):

        _img_dataset = []
        _pnet_boxes = nms_utils.convert_to_square(pnet_boxes) #转成正方形
        # print("_pnet_boxes==>",_pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))                #符合rnet的输入形状 24*24
            img_data = self.image_transfrom(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset) #做堆叠成批次
        # print("_img_dataset==>",_img_dataset)

        if self.iscuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)              #难点 维度变换
        # print("_cls",_cls,"_offset",_offset)
        cls = _cls[:,0,0].cpu().data.numpy()              #把维度上元素为一的维度去掉
        offset = _offset[:,:,0].cpu().data.numpy()        #.cpu().data.numpy()???
        # landmarks = _landmarks[:,:,0].cpu().data.numpy()
        # cls = _cls[0][0].cpu().data.numpy()          #原代码此处是全链接
        # offset = _offset[0].cpu().data.numpy()
        # print("rnetcls==>",cls,"offset",offset)
        # print("rnetcls==>",cls)
        #cls=[cls1,cls2..]  offset =[[offx1,offy1,offx2,offy2]]
        # print(type(cls))
        boxes = []
        idxs = np.where(cls > 0.7)    #置信度大于0.8 的框 索引  idxs = (array([4, 5]),)
        # print(idxs[0])
        # print("idxs==>",cls[idxs])
        for idx in idxs[0]:                #反算坐标
            _box = _pnet_boxes[idx]
            # print("idx",idx)
            # print("_pnet_boxes[idx]",_pnet_boxes[1])
            _x1 = int(_box[0])      #pnet_boxes 坐标only size-1 arrays can be converted to Python scalars
            _y1 = int(_box[1])      ## Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
            _x2 = int(_box[2])
            _y2 = int(_box[3])


            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0][0]
            y1 = _y1 + oh * offset[idx][1][0]
            x2 = _x2 + ow * offset[idx][2][0]
            y2 = _y2 + oh * offset[idx][3][0]
            # print("offset[idx][3]==>",offset[idx][3][0])

            boxes.append([x1, y1, x2, y2, cls[idx]])
            # print("boxes",boxes)
        return nms_utils.nms(np.array(boxes), thresh = 0.6)   #ismin =True

    def onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = nms_utils.convert_to_square(rnet_boxes)
        # print("_rnet_boxes==>",_rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.image_transfrom(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        # print("img_data==>",img_data)
        if self.iscuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset,_landmarks = self.onet(img_dataset)
        # print("_cls==>", _cls, "_offset", _offset)

        cls = _cls[:, 0, 0].cpu().data.numpy()  # 把维度上元素为一的维度去掉
        offset = _offset[:, :, 0].cpu().data.numpy()
        landmarks = _landmarks[:,:,0].cpu().data.numpy()

        # cls = _cls.squeeze().cpu().data.numpy()     #两种写法在不止一个框情况下同样效果，当在只有一个框情况下报错
        # offset = _offset.squeeze().cpu().data.numpy()#原因在offset[idx][0]，offset必须是二维 offset=[[x1,x2,y1,y2]]
        # print("cls==>", cls, "offset", offset,"landmarks",landmarks)

        # cls = _cls.cpu().data.numpy()
        # offset = _offset.cpu().data.numpy()
        boxes = []
        idxs = np.where(cls > 0.9)
        # print(idxs)
        # print("idex[0]==>",idxs[0])
        for idx in idxs[0]:
            # print(idx)#注意  idxs= (array([]),)
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0][0]
            y1 = _y1 + oh * offset[idx][1][0]
            x2 = _x2 + ow * offset[idx][2][0]
            y2 = _y2 + oh * offset[idx][3][0]

            lefteye_x = landmarks[idx][0][0] * ow + _x1
            lefteye_y = landmarks[idx][1][0] * oh + _y1
            righteye_x = landmarks[idx][2][0] * ow + _x1
            righteye_y = landmarks[idx][3][0] * oh + _y1
            nose_x = landmarks[idx][4][0] * ow + _x1
            nose_y = landmarks[idx][5][0] * oh + _y1
            leftmouth_x = landmarks[idx][6][0] * ow + _x1
            leftmouth_y = landmarks[idx][7][0] * oh + _y1
            rightmouth_x = landmarks[idx][8][0] * ow + _x1
            rightmouth_y = landmarks[idx][9][0] * oh + _y1

            boxes.append([x1, y1, x2, y2, cls[idx],lefteye_x, lefteye_y ,righteye_x ,righteye_y, nose_x, nose_y,leftmouth_x, leftmouth_y ,rightmouth_x ,rightmouth_y])

        return nms_utils.nms(np.array(boxes), 0.7, isMin=True)
    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2.0, side_len=12.0):  #start_index 为list结构【544,731】HW结构索引
        # print(start_index, offset, cls, scale, stride, )
        # print("start_index[1]==>",start_index[1].type())
        # print(stride.type())
        _x1 = ((start_index[1].float() * stride) / scale)#start_index[1].float()  注意与—x1.float（)的区别,这样第二次循环会出现  136 错误
        _y1 = ((start_index[0].float() * stride) / scale)


        # print("_y1==>", _y1, _y1.type())
        _x2 = ((start_index[1].float() * stride + side_len) / scale)
        _y2 = ((start_index[0].float() * stride + side_len) / scale)

        ow = (_x2 - _x1)
        oh = (_y2 - _y1)        #ow = oh ,原因是因为生成时把样本框变成正方形了
        # print(ow)
        _offset = offset[:, start_index[0], start_index[1]]  #offset为TENSOR【【【  】】】CHW结构
        # _landmarks = landmarks[:, start_index[0], start_index[1]]
        # print("_offset==>",_offset)   #[ 0.0411, -0.1196, -0.9950,  0.1096]
        # print(_offset[0].type())
        # print("_x1.type==>",_x1.type())
        x1 = _x1 + ow * _offset[0] #Expected object of type torch.LongTensor but found type torch.FloatTensor for argument #2 'other'
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]
if __name__ == '__main__':

    # image = Image.open(os.path.join(test_image_path, filename))
    image_path = r"/home/tensorflow01/oneday/celeba/test_image"
    detector = detector()
    for epoch in range(6,7):

        print(epoch)
        img = Image.open(os.path .join(image_path,"{0}.jpg".format(epoch)))
        print(img)
        print(type(img))
        print(img.size)
        boxes = detector.detect(img)
        # print("out_boxes==>", boxes)
        imDraw = ImageDraw.Draw(img)
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
            print("cls==>",box[4])
            imDraw.rectangle((x1, y1, x2, y2), outline='red',width=1)
            imDraw.ellipse([lefteye_x, lefteye_y,lefteye_x+5, lefteye_y+5],fill="blue")
            imDraw.ellipse([righteye_x, righteye_y,righteye_x+5, righteye_y+5],fill="blue")
            imDraw.ellipse([nose_x, nose_y,nose_x+5, nose_y+5],fill="blue")
            imDraw.ellipse([leftmouth_x,leftmouth_y,leftmouth_x+5,leftmouth_y+5],fill="blue")
            imDraw.ellipse([rightmouth_x,rightmouth_y,rightmouth_x+5,rightmouth_y+5],fill="blue")
            # imDraw.polygon([(lefteye_x, lefteye_y), (righteye_x, righteye_y), (nose_x, nose_y),
            #                 (leftmouth_x, leftmouth_y), (rightmouth_x, rightmouth_y)], fill=(255, 0, 0))
            # print((x1, y1, x2, y2))
        img.show()


