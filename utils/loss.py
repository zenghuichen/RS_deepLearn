'''


'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    FocalLoss 损失函数，这里使用了nlloss 作为基础实现组件
    '''
    def __init__(self,weight,gamma,reduction='mean'):
        super(FocalLoss,self).__init__()
        self.weight=weight
        self.gamma=gamma
        self.loss=nn.NLLLoss(weight=self.weight,reduction=reduction)

    def forward(self,inputs,gt):
        '''
        假定input没有执行softmax函数
        '''
        inputs=F.softmax(inputs)
        loss_value=self.loss((1-inputs)**2*torch.log(inputs),gt)
        return loss_value 

def get_lossfunction(config_param):

    #lossobj = SegmentationLosses(weight=config_param['lossfunction']["weight"],size_average=False)
    # lossobj.build_loss(config_param['lossfunction']["name"])
    return  FocalLoss(weight=config_param["lossfunction"]['weight'],
                        gamma=config_param['lossfunction']['gamma'],
                        reduction=config_param['lossfunction']['reduction'])# CrossEntropyLoss(weight=config_param['lossfunction']["weight"])