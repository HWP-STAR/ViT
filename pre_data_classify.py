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
    transform=transforms.Compose([
            transforms.Resize((size,size)),
            GrayToRgb(),
            transforms.ToTensor()
        ])

    train_dataset=torchvision.datasets.MNIST(
        root='~/data',download=False,
        transform=transform,train=True
            )
    test_dataset=torchvision.datasets.MNIST(
        root='~/data',train=False,download=False,transform=transform
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
        root='~/data',train=True,download=True,transform=transform
            )
    test_dataset=torchvision.datasets.CIFAR10(
        root='~/data',train=False,download=True,transform=transform
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
    # 1. 修复语法错误：补全Compose的闭合括号，格式化代码提升可读性
    transform = transforms.Compose([
        transforms.Resize((size, size)),  # 调整图像尺寸为指定大小
        transforms.ToTensor(),  # 转换为Tensor格式
        # 标准化：保持原有的均值和标准差参数，补全闭合括号
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2345, 0.2616]
        )
    ])

    # 2. 关键修复：OxfordIIITPet不支持train参数，替换为官方支持的split参数
    # 训练集+验证集：使用split='trainval'（官方标准划分）
    train_dataset = torchvision.datasets.OxfordIIITPet(
        root='~/data',
        split='trainval',  # 替换原有的train=True
        target_types='category',  # 明确指定分类任务（默认也是该值，显式声明更清晰）
        download=False,
        transform=transform
    )

    # 测试集：使用split='test'（官方标准划分）
    test_dataset = torchvision.datasets.OxfordIIITPet(
        root='~/data',
        split='test',  # 替换原有的train=False
        target_types='category',  # 明确指定分类任务
        download=False,
        transform=transform
    )

    # 训练集数据加载器（保持原有配置，优化num_workers兼容性）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # 使用适配后的工作线程数
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2 
    )

    # 测试集数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # 使用适配后的工作线程数
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2 
    )

    # 4. 优化打印信息：更清晰的数据集描述
    print(f'train数据集大小：{len(train_dataset)}')
    print(f'分类类别总数：{len(train_dataset.classes)}')  # 训练集和测试集类别一致，无需重复打印
    print(f'test数据集大小：{len(test_dataset)}')
    # 可选：打印前5个类别名称，方便验证
    print(f'前5个类别名称：{train_dataset.classes[:5]}')

    # 5. 修复语法错误：将中文逗号替换为英文逗号
    return train_loader, test_loader

#cifar100数据集

def cifar100_loader(size=32,batch_size=64,num_workers=16):

    transform =transforms.Compose([
            transforms.Resize((size,size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2470,0.2345,0.2616])]#标准化
            
            )

    train_dataset=torchvision.datasets.CIFAR100(
        root='~/data',train=True,download=True,transform=transform
            )
    test_dataset=torchvision.datasets.CIFAR100(
        root='~/data',train=False,download=True,transform=transform
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
