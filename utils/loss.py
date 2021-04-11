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
        
        self.gamma=gamma
        self.reduction=reduction
        if weight is None:
            self.weight=None
        else:
            self.weight=torch.tensor(weight).cuda()
        
    def forward(self,inputs,gt):
        '''
        假定input没有执行softmax函数
        '''
        inputs=F.softmax(inputs) # 求解分布概率
        loss_tatal=0
      
        for i in range(gt.shape[1]):
            loss_tatal=loss_tatal-self.weight[i]*torch.pow(1-inputs[:,i,:,:],self.gamma)*torch.log10(inputs[:,i,:,:])*gt[:,i,:,:]    # Focal Loss 累加方法 --直接构建公式
        loss_value=torch.sum(loss_tatal)/100
        
        if self.reduction=='none':
            return torch.sum(loss_value)
        return loss_value 

class CrossEntropyLoss2d(nn.Module):
    def __init__(self,weight,reduction='mean'):
        super(CrossEntropyLoss2d,self).__init__()
        self.weight=torch.tensor(weight).cuda()
        self.reduction=reduction
        self.loss=nn.CrossEntropyLoss(self.weight,reduction=reduction)

    def forward(self,inputs,gt):
        return self.loss(inputs,gt)


def dice_coeff(pred, target):
    '''
    pred 预测值 [N,C,H,W]
    target 数据值 [N,C,H,W]
    '''
    smooth = 1.0
    sum1=pred*target
    sum2=pred+target
    sum1=2*torch.sum(sum1)
    sum2=torch.sum(sum2)
    return (sum1+smooth)/(sum2+smooth)

class SoftDiceLoss(nn.Module):
    '''
    Dice Loss 
    '''
    def __init__(self, weight=None, reduction="mean"):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        logits=F.sigmoid(logits)
        return 1-dice_coeff(logits,targets)


class BCELoss2d(nn.Module):
    '''
    BCE Loss 
    '''
    def __init__(self, weight=None, reduction="mean"):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction=reduction)
 
    def forward(self, logits, targets):
        probs = F.sigmoid(logits)[1]  # 二分类问题，sigmoid等价于softmax
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)

def get_lossfunction(config_param):

    #lossobj = SegmentationLosses(weight=config_param['lossfunction']["weight"],size_average=False)
    # lossobj.build_loss(config_param['lossfunction']["name"])
    
    return  FocalLoss(weight=config_param["lossfunction"]['weight'],
                        gamma=config_param['lossfunction']['gamma'],
                        reduction=config_param['lossfunction']['reduction']) # CrossEntropyLoss(weight=config_param['lossfunction']["weight"])
    
    #return CrossEntropyLoss2d(weight=config_param["lossfunction"]['weight'])
    #return SoftDiceLoss(weight=config_param["lossfunction"]['weight'],reduction=config_param['lossfunction']['reduction'])
    #return nn.NLLLoss().cuda()
