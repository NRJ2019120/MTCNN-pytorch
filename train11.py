import torch
import os
import torch.nn as nn
import sampling as samp
import torch.utils.data as data
import numpy as np

class Trainer(nn.Module):   #nn.Module 所有模型的父类

    def __init__(self,net,parampath,rootpath,isCuda):
        super(Trainer,self).__init__()
        self.net = net
        self.parampath = parampath
        self.rootpath = rootpath
        self.isCuda = isCuda
        self.con_lossfun = nn.BCELoss()  # 置信度损失函数     二元交叉熵损失 Binary Cross Entropy
        self.offset_lossfun = nn.MSELoss()  # 坐标的损失函数w  均方差损失 mean squared error
        self.landmarks_lossfun = nn.MSELoss()
        if self.isCuda == True:
            self.net.cuda()
    def train(self):

        dataset = samp.FaceDataset(self.rootpath)
        self.dataloader = data.DataLoader(dataset, batch_size=512, shuffle=True,num_workers=4)
        self.optimer = torch.optim.Adam(self.net.parameters(),lr=1e-4,weight_decay=0.00005)#weight_decay  正则化因子

        if os.path.exists(self.parampath):  # 恢复参数继续训练
            self.net.load_state_dict(torch.load(self.parampath))
        if not os.path .exists(self.parampath): #创建参数文件
            dirs,file = os.path.split(self.parampath)
            if not os.path.exists(dirs):
                os.makedirs(dirs)
        #训练
        while True:
           for epoch,(imgdata, con, offset) in enumerate(self.dataloader):  # batchsize
            if self.isCuda == True:              #GPU 运算
                imgdata, con, offset = imgdata.cuda(), con.cuda(), offset.cuda()#,landmarks.cuda()

            con_out, offset_out = self.net(imgdata)   # imgdata为网络输入, 得到网络输出
             #如何保证con_out  > 0???        # Assertion `x >= 0. && x <= 1.' failed. input value should be between 0~1, but got -0.157099 at
                                             # 加signmod 函数 转化成概率
            # 计算置信度损失
            # print(con)
            # print("==================")
            # print(con_out)
            con_mask = torch.lt(con,2)          #部分样本不参与分类损失   2定义为部分样本  掩码操作
            con_ = con[con_mask[:,0]]
            # print("con_",con_.shape)
            con_out_ = con_out[con_mask[:,0]]
            con_loss = self.con_lossfun(con_out_,con_)  #只有正负样本参与置信度损失计算

            #计算坐标偏移损失
            offset_mask = torch.gt(con, 0)             #选出非负样本
            offset_ = offset[offset_mask[:,0]]
            offset_out_ = offset_out[offset_mask[:,0]]
            offset_loss = self.offset_lossfun(offset_out_, offset_)  # 只有非负样本参与坐标偏移损失计算

            # landmarks_ = landmarks[offset_mask[:,0]]            #con>0 包含正样本和正样本
            # # print(landmarks_.shape)
            # landmarks_out_ = landmarks_out[offset_mask[:, 0]]   #con>0 包含正样本和正样本
            #
            # landmarks_mask = torch.lt(offset_,2)                   # 只选正样本 con >0  and con <2
            # landmarks_ = landmarks_[landmarks_mask[:,0]]
            # landmarks_out_ = landmarks_out_[landmarks_mask[:, 0]]
            # # print(landmarks_mask)
            # # landmarks_ = landmarks[offset_mask[:,0]]
            # # landmarks_out_ = landmarks_out[offset_mask[:,0]]
            # landmarks_loss = self.landmarks_lossfun(landmarks_out_,landmarks_)
            #总loss
            total_loss = con_loss + offset_loss #+ landmarks_loss

            self.optimer.zero_grad()
            total_loss.backward()
            self.optimer.step()
            # break
            if epoch % 100 == 0 and epoch>0:
                torch.save(self.net.state_dict(),self.parampath)  #保存模型参数
                print("epoch==>",epoch,"total_loss==>",total_loss)
            # if epoch % 10000 == 0:
            #     feed = input("whether continue or over train YES OR NO")
            #     if feed == "YES":
            #         continue
            #     else:
            #         break

# if __name__ == '__main__':
#     net = MTCNNnet.P_NET()
#     net.train()  # 参数参与更新,如果是测试使用则是 net.eval().  r"Sets the module in training mode".
#
#     onet_trainer = Trainer(net,P_parampath,root_path12,isCuda=False)
#     onet_trainer.train()





