import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self,path,island=True):

        self.island = island
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path,"nonlandmarks_positive.txt")).readlines()) #列表增添,正 ,负,部分样本标签聚集在dataset中
        self.dataset.extend(open(os.path.join(path,"nonlandmarks_negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path,"nonlandmarks_part.txt")).readlines())

    def __getitem__(self, index):
        "{0}.jpg 置信度 x1 y1 x2 y2"
        strs = self.dataset[index].strip().split()
        # print(strs)    # 样本标签
        confidence = torch.Tensor([float(strs[1])])  # 置信度   用float格式  ,谨慎用int
        # print(confidence)                         #tensor([1.4714e-43])==tensor([1])
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])  # 偏移量
        # if self.island:
        #     landmarks = torch.Tensor([float(strs[6]),float(strs[7]),float(strs[8]),float(strs[9]),float(strs[10]),float(strs[11]),
        #                          float(strs[12]),float(strs[13]),float(strs[14]),float(strs[15])])
        # print(offset)

        if strs[1] == '1':  # 正样本
            img_data = torch.Tensor(np.array(Image.open(os.path.join(self.path,"positive",strs[0])), dtype=np.float32) /255-0.5)
        elif strs[1] == '0':  # 负样本
            img_data = torch.Tensor(np.array(Image.open(os.path.join(self.path,"negative",strs[0])), dtype=np.float32) /255-0.5)
        elif strs[1] == '2':  # 部分样本
            img_data = torch.Tensor(np.array(Image.open(os.path.join(self.path,"part",strs[0])), dtype=np.float32) /255-0.5)

        img_data = img_data.transpose(2,0)        #HWC-->CWH
        img_data = img_data.transpose(2,1)        #HWC-->CHW  [12,12,3]or[24,24,3]or[48,48,3]

        return img_data,confidence,offset

    def __len__(self):

        return len(self.dataset)
"""
if __name__ == '__main__':
#
    root_path12 =r'/media/tensorflow01/myfile/landmarks_img_celeba/12'
    dataset_12 = FaceDataset(root_path12)
#     # print(len(dataset_12))
#     # print(dataset_12[0])  # 通道是3 imgdata ,con,offset
#     # print(dataset_12)      #_<_main__.FaceDataset object at 0x00000000023C5CF8>
#     # #
    dataloader = data.DataLoader(dataset_12,batch_size=4,shuffle=True)
    for epoch,(imgdata,con ,offset,landmaeks) in enumerate(dataloader):
        print("epoch==>", epoch, "imgdata==>", imgdata, "con==>", con, "offset==>", offset)
        # con_onehot = con2onehot.to_onehot(con)
        # print(con_onehot)
        break # imgdata==> NCHW ;CON-->NV  ;offset-->NV
#     # for imgdata,con ,offset in dataloader:
    #     print("imgdata==>",imgdata,"con==>",con,"offset==>",offset)
    #     break"""


