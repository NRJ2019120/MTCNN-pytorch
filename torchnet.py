import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):                #参数初始化
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform(m.weight)
        nn.init.constant(m.bias,0.1)
class P_NET(nn.Module):

    def __init__(self,istraining = True):
        super(P_NET,self).__init__()
        self.istraining = istraining
        self.conv1 = nn.Sequential(
             nn.Conv2d(3,10,kernel_size=3,stride=1)
             ,nn.PReLU()
            ,nn.MaxPool2d(kernel_size=2,stride=2)  #5*5*10
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,16,kernel_size=3,stride=1) #3*3*16
            ,nn.PReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1) #1*1*32
           ,nn.PReLU()
            )
        self.confident = nn.Conv2d(32,1,kernel_size=1,stride=1)#置信度  双头龙输出
        self.offset = nn.Conv2d(32,4,kernel_size=1,stride=1)#人脸框坐标偏移
        # self.landmarks = nn.Conv2d(32,10,kernel_size=1,stride=1)

        self.apply(init_weights) #权重初始化

    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        con = F.sigmoid(self.confident(conv3))#置信度
        offset = self.offset(conv3) #偏移
        # landmarks =self.landmarks(conv3)

        if self.istraining == True:           #理解了吗？
            con = con.view(-1,1)   #NV
            offset = offset.view(-1,4)# NV
            # landmarks = landmarks.view(-1,10) #NV
        return con, offset
class R_NET(nn.Module):

    def __init__(self,istraining = True):
        super(R_NET,self).__init__()

        self.istraining = istraining
        self.conv1 = nn.Sequential(             #输入24*24*3
            nn.Conv2d(3,28,kernel_size=3,stride=1)#用3*3
            ,nn.PReLU()
            ,nn.MaxPool2d(3,stride=2,padding=1) #11*11*28 此处需要用padding
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(28,48,kernel_size=3,stride=1)
            ,nn.PReLU()
            ,nn.MaxPool2d(3,stride=2)            #4*4*48
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48,64,kernel_size=2,stride=1)#3*3*64
            ,nn.PReLU()
        )
        self.confidence = nn.Conv2d(64,1,kernel_size=3,stride=1)#1*1*1
        self.offset = nn.Conv2d(64,4,kernel_size=3,stride=1)#1*1*4
        # self.landmarks = nn.Conv2d(64, 10, kernel_size=3, stride=1)
        self.apply(init_weights)

    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        con = F.sigmoid(self.confidence(conv3))  # 置信度
        offset = self.offset(conv3)
        # landmarks = self.landmarks(conv3)

        if self.istraining == True:
            con = con.view(-1,1)          #形状变换NV
            offset = offset.view(-1,4)
            # landmarks = landmarks.view(-1, 10)

        return con, offset

class O_NET(nn.Module):
    def __init__(self,istraining=True):
        super(O_NET, self).__init__()

        self.istraining = istraining
        self.conv1 = nn.Sequential(                   #输入48*48*3
            nn.Conv2d(3, 32, kernel_size=3, stride=1)
            , nn.PReLU()
            , nn.MaxPool2d(3, stride=2,padding=1)     #23*23*32 需要用padding
            , nn.Conv2d(32, 64, kernel_size=3, stride=1)
            , nn.PReLU()
            , nn.MaxPool2d(3, stride=2)                 # 10*10*64
            , nn.Conv2d(64, 64, kernel_size=3, stride=1)
            , nn.PReLU()
            , nn.MaxPool2d(2, stride=2)                   #4*4*64
            , nn.Conv2d(64, 128, kernel_size=2, stride=1) # 3*3*128
            , nn.PReLU()
            )

        self.confidence = nn.Conv2d(128, 1, kernel_size=3, stride=1)
        self.offset = nn.Conv2d(128, 4, kernel_size=3, stride=1)
        self.landmarks = nn.Conv2d(128, 10, kernel_size=3, stride=1)

        self.apply(init_weights)#权重初始化

    def forward(self,x):
        conv1_out = self.conv1(x)
        con = F.sigmoid(self.confidence(conv1_out))  # 置信度
        offset = self.offset(conv1_out)     #偏移
        landmarks = self.landmarks(conv1_out)
        if self.istraining == True:     #训练
            con = con.view(-1, 1)       #NCHW==>NV  形状变换
            offset = offset.view(-1, 4) #NCHW==>NV
            landmarks = landmarks.view(-1, 10)

        return con, offset,landmarks


