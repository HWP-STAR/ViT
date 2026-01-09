import torch
import torch.nn as nn
import torch.optim as optim

#自己的工具库
from pre_data_classify import cifar10_loader,oxfordIIIPet_loader,cifar100_loader
from tools_classify import train_basic,evaluate

#编写的模型
#from models_experiment_classify import myCNN,myCNN_2,ResNet50
#from my_try import SimpleResNet,my_GoogLeNet,SimpleSeNet,SimpleMobileNetV1
#from temp import ViT
from models import ViT
#from multilayCNN import MultiLayCNN,oneCNN,ThreeLinear

#=====模型训练=====
device=torch.device('cuda:0')
print(f'使用设备：{device}')


#数据加载
size=64
in_channels=3

batch_size=64
num_workers=16
train_loader,test_loader=cifar10_loader(size=size,batch_size=batch_size,num_workers=num_workers)

model=ViT(num_classes=10,image_size=size,
            patch_size=4,dim=128,depth=12,heads=4,mlp_dim=256,
            dropout=0.1,emb_dropout=0.1
        ).to(device)
#model=MultiLayCNN().to(device)

cri=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-4)

train_basic(model,train_loader,cri,opt,device,batch_size,epochs=3)
evaluate(model,test_loader,cri,device)

print('train数据集再次验证')
evaluate(model,train_loader,cri,device)
