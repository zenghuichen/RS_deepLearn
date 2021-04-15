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
        sample['source_image']=np.copy(image)
        return sample

class RandomFlip:
    '''
    随机翻转
    '''
    def __call__(self, sample):
        image, seg, label,source_image = sample['img'], sample['seg'], sample['label'],sample['source_image']
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            source_image=np.fliplr(source_image).copy() # 方便查看原图
            seg = np.fliplr(seg).copy()
            label = np.fliplr(label).copy()

        sample['img'] = image
        sample['seg'] = seg
        sample['label'] = label
        sample['source_image']=source_image
        return sample

class RandomRotate:
    '''
    随机旋转图像
    随机旋转图像应在随机裁剪之前
    如果没有无损图像旋转方法，则不应启用这个方法
    '''
    def __call__(self,sample):
        image, seg, label,source_image = sample['img'], sample['seg'], sample['label'],sample['source_image']
        # 
        sample['img']=image
        sample['seg']=seg
        sample['label']=label
        sample['source_image']=source_image
        return sample
class Normalize:
    '''
    数据归一化，将数据归一化到的[-1,1]之间，平均值为0

    '''
    def __init__(self,meanArr,stdArr,VI_enable=False):
        self.mean_arr=np.array(meanArr)
        self.std_arr= np.array(stdArr)
        self.VI_enable=VI_enable

    def __call__(self, sample):
        if self.VI_enable: # 指数归一化
            return sample
        else:
            image, seg, label,source_image = sample['img'], sample['seg'], sample['label'],sample['source_image']
            image=(image-self.mean_arr)/self.std_arr # z-score 零-均值归一化
            # 数据正则化 
            # 这里假设当前平均值为4000， 最大值为+_20000
            #image=(image-self.mean_arr)*0.01
            #image=(image-4000)/20000 
            sample['img']=image
            sample['seg']=seg
            sample['label']=label
            sample['source_image']=source_image
            return sample

class BandVI:
    '''
    进行波段组合运算，注意这个方法只有在allbands时，可以启用
    '''
    def __init__(self,enable,minvalue,maxvalue):
        self.enable=enable
        self.minvalue=minvalue
        self.maxvalue=maxvalue
    
    def __call__(self,sample):
        if not self.enable:
            return sample
        bandimg=sample['img']
        bands=bandimg.shape[2] # 假定此方法在维度转换之前
        VI_num=int(bands*(bands-1)/2)
        band_VI=np.zeros((bandimg.shape[0],bandimg.shape[1],VI_num),dtype=np.float32)
        
        # 缩放数据空间
        bandimg=bandimg-self.minvalue
        t=0
        for i in range(bands):
            for j in range(i+1,bands):
                bandVI_deno=bandimg[:,:,i]+bandimg[:,:,j]
                bandVI_num=bandimg[:,:,i]-bandimg[:,:,j]
                band_VI[:,:,t]=bandVI_num/(bandVI_deno+1)
                t=t+1
        sample['img']=band_VI
        sample['source_image']=(band_VI[:,:,:3].copy()+1)/2
        return sample

class OneHot:
    '''
    one-hot encoding 方法，方便计算损失函数
    '''
    def __init__(self,cls_num):
        self.cls_num=cls_num
        pass
    def __call__(self,sample):
        image, seg, label,source_image = sample['img'], sample['seg'], sample['label'],sample['source_image']
        new_label=np.zeros((self.cls_num,label.shape[0],label.shape[1]),dtype=np.uint8)
        for i in range(0,self.cls_num):
            new_label[i,:,:]=(label==i)*1
        sample['img']=image
        sample['seg']=seg
        sample['label']=new_label    
        sample['source_image']=source_image    
        return sample

class ToTensor:
    def __init__(self,driver='cpu'):
        self.driver=driver
    def __call__(self,sample):
        image, seg, label,source_image = sample['img'], sample['seg'], sample['label'],sample['source_image']
        sample['img']= torch.from_numpy(np.transpose(image,(2,0,1))).float()
        sample["source_image"]=torch.from_numpy(np.transpose(source_image,(2,0,1))).float()
        sample['seg']=torch.from_numpy(seg).long()
        sample['label']=torch.from_numpy(label).long()
        if self.driver=='cuda':
            sample['img']=sample['img'].cuda()
            sample["source_image"]=sample["source_image"].cuda()
            sample['seg']=sample['seg'].cuda()
            sample['label']=sample['label'].cuda()
        return sample
