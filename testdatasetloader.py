
import os 
print(os.getcwd())

#import cv2
import matplotlib
import matplotlib.colors
import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision
import torchvision.transforms as transforms

from datasetloader.get_datasetLoader import *

if __name__=="__main__":
    E_train_loader,E_val_loader,E_test_loader=get_dataset(datasetName="RGB432",num_workers=1)
    for i, sample in tqdm(enumerate(E_train_loader), total = len(E_train_loader)):
        image,seg,label=sample['img'], sample['seg'], sample['label']
    