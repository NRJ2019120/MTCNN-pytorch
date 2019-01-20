import torchnet as MTCNNnet
from train11 import Trainer as Trainer


O_parampath = r"/home/tensorflow01/oneday/celeba/landmarks_params/onet.pkl"
root_path12 = r'/media/tensorflow01/myfile/landmarks_img_celeba/12'
root_path24 = r'/media/tensorflow01/myfile/landmarks_img_celeba/24'
root_path48 = r'/media/tensorflow01/myfile/landmarks_img_celeba/48'
#
if __name__ == '__main__':
    net = MTCNNnet.O_NET()
    net.train()  # 参数参与更新,如果是测试使用则是 net.eval().  "Sets the module in training mode".
    onet_trainer = Trainer(net,O_parampath,root_path48,isCuda=True)
    onet_trainer.train()