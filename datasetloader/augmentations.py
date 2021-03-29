'''
图像预处理模块
This code is partially adapted from ESANet
'''
#import cv2
import matplotlib
import matplotlib.colors
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.preprocessing import normalize
class RandomCrop:
    '''
    随机裁剪
    '''
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample):
        image, seg, label = sample['img'], sample['seg'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = np.random.randint(0, h - self.crop_height) if h>self.crop_height else 0
        j = np.random.randint(0, w - self.crop_width) if w > self.crop_height else 0
        image = image[i:i + self.crop_height, j:j + self.crop_width, :]
        seg = seg[i:i + self.crop_height, j:j + self.crop_width]
        label = label[i:i + self.crop_height, j:j + self.crop_width]
        sample['img'] = image
        sample['seg'] = seg
        sample['label'] = label
        return sample

class RandomFlip:
    '''
    随机翻转
    '''
    def __call__(self, sample):
        image, seg, label = sample['img'], sample['seg'], sample['label']
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            seg = np.fliplr(seg).copy()
            label = np.fliplr(label).copy()

        sample['img'] = image
        sample['seg'] = seg
        sample['label'] = label

        return sample

class RandomRotate:
    '''
    随机旋转图像
    随机旋转图像应在随机裁剪之前
    如果没有无损图像旋转方法，则不应启用这个方法
    '''
    def __call__(self,sample):
        image,seg,label=sample['img'], sample['seg'], sample['label']
        # 
        sample['img']=image
        sample['seg']=seg
        sample['label']=label

class Normalize:
    '''
    数据归一化，将数据归一化到的[-1,1]之间，平均值为0

    '''
    def __init__(self,meanArr,stdArr):
        self.mean_arr=np.array(meanArr)
        self.std_arr= np.array(stdArr)
        pass

    def __call__(self, sample):
        image,seg,label=sample['img'], sample['seg'], sample['label']
        image=(image-self.mean_arr)/self.std_arr
        # 数据正则化
        sample['img']=image
        sample['seg']=seg
        sample['label']=label
        return sample


class ToTensor:
    def __init__(self):
        pass
    
    def __call__(self,sample):
        image,seg,label=sample['img'], sample['seg'], sample['label']
        # 
        sample['img']= torch.from_numpy(np.transpose(image,(2,0,1))).float()
        sample['seg']=torch.from_numpy(seg).long()
        sample['label']=torch.from_numpy(label).long()

        return sample