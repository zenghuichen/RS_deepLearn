'''


'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def focalloss(prd,gt):
    '''
    focalloss 在任务上更适合二分类任务
    '''
    


def get_lossfunction(config_param):

    #lossobj = SegmentationLosses(weight=config_param['lossfunction']["weight"],size_average=False)
    # lossobj.build_loss(config_param['lossfunction']["name"])
    return  None# CrossEntropyLoss(weight=config_param['lossfunction']["weight"])