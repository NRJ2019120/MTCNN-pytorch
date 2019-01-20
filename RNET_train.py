import torchnet as MTCNNnet
from train11 import Trainer as Trainer


R_parampath = r"/home/tensorflow01/oneday/celeba/landmarks_params/rnet.pkl"
root_path12 = r'/media/tensorflow01/myfile/landmarks_img_celeba/12'
root_path24 = r'/media/tensorflow01/myfile/landmarks_img_celeba/24'
root_path48 = r'/media/tensorflow01/myfile/landmarks_img_celeba/48'

if __name__ == '__main__':
    net = MTCNNnet.R_NET()
    net.train()  # 参数参与更新,如果是测试使用则是 net.eval().  r"Sets the module in training mode".

    rnet_trainer = Trainer(net,R_parampath,root_path24,isCuda=True)
    rnet_trainer.train()
