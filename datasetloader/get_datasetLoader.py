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
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.sample_output= next(self.loader)
        except StopIteration:
            self.sample_output = None
            return
        '''
        with torch.cuda.stream(self.stream):
            self.sample_output = = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.source_img=self.next_source_img.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_target=self.next_target.long()
            self.next_source_img=self.next_source_img.float()
            #elf.next_input = self.next_input.sub_(self.mean).div_(self.std)
        '''

    def next(self):
        #torch.cuda.current_stream().wait_stream(self.stream)
        sample_output = self.sample_output
        self.preload()
        return sample_output

def get_dataset(datasetName,dataSize='E256',batch_size=4,cropsize=(256,256),num_workers=8):
    '''
    创建对应的数据结构
    datasetName:数据名称 可以选择：RGB123 RGB124 allband RGB134 RGB234
    '''
    dataset_path=getRootdirFromDatasetName(datasetName)
    E_path=dataset_path[dataSize]
    # 处理train训练数据集
    # 构建对应的数据结构
    train_transpose=get_augmentations("train",cropsize,dataset_path,cls_num=dataset_path['cls_num'])
    E_train=gfNanChangDataset(E_path,splitchar='train',augmentations=train_transpose) 
    
    # 处理test数据集
    test_transpose=get_augmentations('test',cropsize,dataset_path,cls_num=dataset_path['cls_num'])
    E_test=gfNanChangDataset(E_path,splitchar='test',augmentations=test_transpose) 

    # 创建对应的加载数据集   
    E_train_loader=torchdata.DataLoader(E_train,batch_size=batch_size,num_workers=num_workers,shuffle=True,drop_last=True)
    E_test_loader=torchdata.DataLoader(E_test,batch_size=batch_size,num_workers=num_workers,shuffle=False,drop_last=True)
    
    return E_train_loader,E_test_loader

def get_augmentations(split,cropsize,E_path,cls_num):
    '''
    split 字符结构
    '''
    if split=="train":
        H_,W_=cropsize
        pre_train_list=[
            RandomCrop(H_,W_),
            RandomFlip(),
            Normalize(E_path["mean"],E_path["std"]),
            OneHot(cls_num),
            ToTensor() ]

        dataset_transpose=transforms.Compose(pre_train_list)
    else:
        H_,W_=cropsize
        pre_val_list=[ 
            #RandomCrop(H_,W_),
            #RandomFlip(),
            Normalize(E_path["mean"],E_path["std"]),
            OneHot(cls_num),
            ToTensor() ]
        dataset_transpose=transforms.Compose(pre_val_list)
    return dataset_transpose



