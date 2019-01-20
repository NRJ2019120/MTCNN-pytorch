import torchnet as MTCNNnet
from train11 import Trainer as Trainer


P_parampath = r"/home/tensorflow01/oneday/celeba/landmarks_params/pnet.pkl"
root_path12 = r'/media/tensorflow01/myfile/landmarks_img_celeba/12'
root_path24 = r'/media/tensorflow01/myfile/landmarks_img_celeba/24'
root_path48 = r'/media/tensorflow01/myfile/landmarks_img_celeba/48'

if __name__ == '__main__':
    net = MTCNNnet.P_NET()
    net.train()  # 参数参与更新,如果是测试使用则是 net.eval().  r"Sets the module in training mode".

    pnet_trainer = Trainer(net,P_parampath,root_path12,isCuda=True)
    pnet_trainer.train()
    # epoch==> 0 loss==>tensor(0.8946, device='cuda:0', grad_fn=<ThAddBackward>)
