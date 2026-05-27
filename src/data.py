#图像分类数据加载

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class GrayToRgb():
    def __call__(self,img):
        return img.convert('RGB')

def mnist_loader(size=32,batch_size=64,num_workers=16):
    transform=transforms.Compose([ transforms.Resize((size,size)),
            GrayToRgb(),
            transforms.ToTensor()
        ])

    train_dataset=torchvision.datasets.MNIST(
        root='../../data',download=False,
        transform=transform,train=True
            )
    test_dataset=torchvision.datasets.MNIST(
        root='../../data',train=False,download=False,transform=transform
            )
   #数据加载器

    train_loader=DataLoader(
        train_dataset,batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,pin_memory=True,prefetch_factor=2
           )

    test_loader=DataLoader(
        test_dataset,batch_size=batch_size,num_workers=num_workers,
        shuffle=False,pin_memory=True,prefetch_factor=2
           )
    print(f'train数据集大小：{len(train_dataset)}')
    print(f'train类数量：{len(train_dataset.classes)}')
    print(f'test数据集大小：{len(test_dataset)}')
    print(f'test类数量：{len(test_dataset.classes)}')
    
    return train_loader,test_loader

#cifar10数据集

def cifar10_loader(size=32,batch_size=64,num_workers=16):

    transform =transforms.Compose([
            transforms.Resize((size,size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2470,0.2345,0.2616])]#标准化
            
            )

    train_dataset=torchvision.datasets.CIFAR10(
        root='../../data',train=True,download=True,transform=transform
            )
    test_dataset=torchvision.datasets.CIFAR10(
        root='../../data',train=False,download=True,transform=transform
            )


    train_loader=torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size,num_workers=num_workers,
        shuffle=True,pin_memory=True,prefetch_factor=2
            )

    test_loader=torch.utils.data.DataLoader(
        test_dataset,batch_size=batch_size,num_workers=num_workers,
        shuffle=False,pin_memory=True,prefetch_factor=2
            )
    print(f'train数据集大小：{len(train_dataset)}')
    print(f'train类别：{len(train_dataset.classes)}')
    print(f'test数据集大小：{len(test_dataset)}')
    print(f'test类别：{len(test_dataset.classes)}')

    return train_loader,test_loader


def oxfordIIIPet_loader(size=224, batch_size=64, num_workers=16):
    # ImageNet 标准化参数（OxfordIIITPet 是自然图像）
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # 训练集：带数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # 测试集：仅缩放
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_dataset = torchvision.datasets.OxfordIIITPet(
        root='../../data',
        split='trainval',
        target_types='category',
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.OxfordIIITPet(
        root='../../data',
        split='test',
        target_types='category',
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2
    )

    print(f'train数据集大小：{len(train_dataset)}')
    print(f'分类类别总数：{len(train_dataset.classes)}')
    print(f'test数据集大小：{len(test_dataset)}')
    print(f'前5个类别名称：{train_dataset.classes[:5]}')

    return train_loader, test_loader

#cifar100数据集

def cifar100_loader(size=32,batch_size=64,num_workers=16):

    transform =transforms.Compose([
            transforms.Resize((size,size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2470,0.2345,0.2616])]#标准化
            
            )

    train_dataset=torchvision.datasets.CIFAR100(
        root='../../data',train=True,download=True,transform=transform
            )
    test_dataset=torchvision.datasets.CIFAR100(
        root='../../data',train=False,download=True,transform=transform
            )


    train_loader=torch.utils.data.DataLoader(
        train_dataset,batch_size=batch_size,num_workers=num_workers,
        shuffle=True,pin_memory=True,prefetch_factor=2
            )

    test_loader=torch.utils.data.DataLoader(
        test_dataset,batch_size=batch_size,num_workers=num_workers,
        shuffle=False,pin_memory=True,prefetch_factor=2
            )
    print(f'train数据集大小：{len(train_dataset)}')
    print(f'train类别：{len(train_dataset.classes)}')
    print(f'test数据集大小：{len(test_dataset)}')
    print(f'test类别：{len(test_dataset.classes)}')
    print(f'类名称：{train_dataset.classes[:5]}')

    return train_loader,test_loader


if __name__=='__main__':
    train_loader,test_loader=oxfordIIIPet_loader()
    print('====')
    t,e=cifar10_loader()
    print('===')
    t,e=cifar100_loader()
    print('======')
    t,e=mnist_loader()
