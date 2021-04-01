'''
构建对应的图像数据集
'''
import matplotlib
import matplotlib.colors
import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision
import torchvision.transforms as transforms

from datasetloader.datasetPath import *
from datasetloader.gf2NanChangDataset import gfNanChangDataset
from datasetloader.augmentations import *

def get_dataset(datasetName,dataSize='E256',batch_size=4,cropsize=(256,256),num_workers=8):
    '''
    创建对应的数据结构
    datasetName:数据名称 可以选择：RGB123 RGB124 allband RGB134 RGB234
    '''
    dataset_path=getRootdirFromDatasetName(datasetName)
    E_path=dataset_path[dataSize]
    # 处理train训练数据集
    # 构建对应的数据结构
    train_transpose=get_augmentations("train",cropsize,dataset_path)
    E_train=gfNanChangDataset(E_path,splitchar='train',augmentations=train_transpose) 
    
    # 处理test数据集
    test_transpose=get_augmentations('test',cropsize,dataset_path)
    E_test=gfNanChangDataset(E_path,splitchar='test',augmentations=test_transpose) 

    # 创建对应的加载数据集   
    E_train_loader=torchdata.DataLoader(E_train,batch_size=batch_size,num_workers=num_workers,shuffle=True,drop_last=True)
    E_test_loader=torchdata.DataLoader(E_test,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=True)
    
    return E_train_loader,E_test_loader

def get_augmentations(split,cropsize,E_path):
    '''
    split 字符结构
    '''
    if split=="train":
        H_,W_=cropsize
        pre_train_list=[
            RandomCrop(H_,W_),
            RandomFlip(),
            Normalize(E_path["mean"],E_path["std"]),
            ToTensor() ]

        dataset_transpose=transforms.Compose(pre_train_list)
    else:
        H_,W_=cropsize
        pre_val_list=[ 
            #RandomCrop(H_,W_),
            #RandomFlip(),
            Normalize(E_path["mean"],E_path["std"]),
            ToTensor() ]
        dataset_transpose=transforms.Compose(pre_val_list)
    return dataset_transpose



