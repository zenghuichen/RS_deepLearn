'''
gf2南昌数据的数据集加载类
'''
import numpy as np
import torch
import torch.utils as torchutil
import torchvision
import torch.utils.data as torchdata
import os
import pandas 


class gfNanChangDataset(torchutil.data.Dataset):
    def __init__(self,rootdir,splitchar='train',augmentations=None):
        '''
        rootdir:根目录
        splitchar: 选择对应的数据集
        augmentations : 图像增强方法
        '''
        self.rootdir=os.path.join(rootdir,splitchar)
        self.seg_path=os.path.join(self.rootdir,"seg")
        self.image_path=os.path.join(self.rootdir,"img")
        self.label_path=os.path.join(self.rootdir,"label")
        self.augmentation=augmentations
        # 检索存在的数据集
        seg_list=os.listdir(self.seg_path)
        img_list=os.listdir(self.image_path)
        label_list=os.listdir(self.label_path)
        name_list=set(seg_list).intersection(set(img_list)).intersection(set(label_list))
        self.name_list=list(name_list)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name_=self.name_list[index]
        seg_path_=os.path.join(self.seg_path,name_)
        label_path_=os.path.join(self.label_path,name_)
        img_path_=os.path.join(self.image_path,name_)
        seg_=np.load(seg_path_).astype(np.float)
        label_=np.load(label_path_).astype(np.float)
        img_=np.load(img_path_).astype(np.float)
        # 数据预处理

        if self.augmentation is None:
            return {'img':img_,"seg":seg_,"label":label_,"name":name_}
        output_= {'img':img_,"seg":seg_,"label":label_,"name":name_}
        output_=self.augmentation(output_)
        return output_













