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


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        if sys.platform=='win32':
            self.preload()
        elif sys.platform=='linux':
            pass

    def preload(self):
        try:
            self.sample_output= next(self.loader)
        except StopIteration:
            self.sample_output = None
            return None
        #with torch.cuda.stream(self.stream):
        #    self.sample_output = self.sample_output.cuda(non_blocking=True)
    def next(self):
        if sys.platform=='win32':
        #torch.cuda.current_stream().wait_stream(self.stream)
            sample_output = self.sample_output
            self.preload()
            return sample_output
        elif sys.platform=='linux':
            try:
                return next(self.loader)
            except StopIteration:
                return None


def get_dataset(config_param,dataSize='E256',cropsize=(256,256)):
    '''
    创建对应的数据结构
    datasetName:数据名称 可以选择：RGB123 RGB124 allband RGB134 RGB234
    '''
    datasetName=config_param["datasetName"]
    batch_size=config_param["batch_size"] 
    num_workers=config_param["num_work"]
    VI_enable= config_param['VI_enable'] if 'VI_enable'  in config_param else False
    
    dataset_path=getRootdirFromDatasetName(datasetName)
    E_path=dataset_path[dataSize]
    # 处理train训练数据集
    # 构建对应的数据结构
    train_transpose=get_augmentations("train",cropsize,dataset_path,cls_num=dataset_path['cls_num'],VI_enable=VI_enable)
    E_train=gfNanChangDataset(E_path,splitchar='train',augmentations=train_transpose) 
    
    # 处理test数据集
    test_transpose=get_augmentations('test',cropsize,dataset_path,cls_num=dataset_path['cls_num'],VI_enable=VI_enable)
    E_test=gfNanChangDataset(E_path,splitchar='test',augmentations=test_transpose) 

    # 创建对应的加载数据集   
    E_train_loader=torchdata.DataLoader(E_train,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=True)
    E_test_loader=torchdata.DataLoader(E_test,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=True)
    
    return E_train_loader,E_test_loader

def get_augmentations(split,cropsize,E_path,cls_num,VI_enable=False):
    '''
    split 字符结构
    '''
    if split=="train":
        H_,W_=cropsize
        if 'min' in E_path:
            pre_train_list=[
                RandomCrop(H_,W_),
                RandomFlip(),
                BandVI(enable=VI_enable,minvalue=E_path['min'],maxvalue=E_path['max']),
                Normalize(E_path["mean"],E_path["std"],VI_enable=VI_enable),
                OneHot(cls_num),
                ToTensor() ]
        else:
                pre_train_list=[
                RandomCrop(H_,W_),
                RandomFlip(),
                #BandVI(enable=VI_enable,minvalue=E_path['min'],maxvalue=E_path['max']),
                Normalize(E_path["mean"],E_path["std"],VI_enable=VI_enable),
                OneHot(cls_num),
                ToTensor() ]

        dataset_transpose=transforms.Compose(pre_train_list)
    else:
        H_,W_=cropsize
        if 'min' in E_path:
            pre_val_list=[ 
                #RandomCrop(H_,W_),
                #RandomFlip(),
                BandVI(enable=VI_enable,minvalue=E_path['min'],maxvalue=E_path['max']),
                Normalize(E_path["mean"],E_path["std"],VI_enable=VI_enable),
                OneHot(cls_num),
                ToTensor() ]
        else:
            pre_val_list=[ 
                #RandomCrop(H_,W_),
                #RandomFlip(),
                #BandVI(enable=VI_enable,minvalue=E_path['min'],maxvalue=E_path['max']),
                Normalize(E_path["mean"],E_path["std"],VI_enable=VI_enable),
                OneHot(cls_num),
                ToTensor() ]
        dataset_transpose=transforms.Compose(pre_val_list)
    return dataset_transpose



