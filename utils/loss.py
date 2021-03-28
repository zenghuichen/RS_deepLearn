import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def get_lossfunction(config_param):
    if config_param['lossfunction']["name"]=='focalloss':
        # {"name":"focalloss","gamma":0, "weight":None, 'size_average':True}
        lossfunction=FocalLoss(gamma=config_param['lossfunction']['gamma'],weight=config_param['lossfunction']['weight'],size_average=config_param['lossfunction']['size_average'])
    
    return lossfunction


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        if weight is None:
            self.loss = nn.NLLLoss( size_average=self.size_average, reduce=False)
        else:
            self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                    size_average=self.size_average, reduce=False)         

    def forward(self, input, target):
        #mask = target > 0
        #targets_m = target.clone()
        #targets_m[mask] -= 1 # 只训练类别为1的对象
        loss_all = self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)
        loss=torch.sum(loss_all)
        return loss
