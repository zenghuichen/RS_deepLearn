'''
Misc Utility functions
'''
from __future__ import division
from collections import OrderedDict
import os
import numpy as np
import pandas as pds
import torch
import sys
def get_scheduler(optimizer,schedulerName="ReduceLROnPlateau",mode="min",patience=10):
    '''
    学习率调整器
    https://blog.csdn.net/weixin_43722026/article/details/103271611
    '''
    if schedulerName=="ReduceLROnPlateau":
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.9, patience=patience, verbose=False, threshold=0.9, threshold_mode='rel', cooldown=0, min_lr=1e-08, eps=1e-08)
    return scheduler

def get_optimizer(model,optimizerName,lr=0.01,momentum=0.9,weight_decay=0):
    '''
    优化器
    '''
    if optimizerName=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    elif optimizerName=="adam":
        optimizer=torch.optim.Adam(model.parameters(), lr=lr,  betas=(0.9, 0.999), eps=1e-8,weight_decay=weight_decay)
    return optimizer


def save_ckpt(ckpt_dir, model, modelName,optimizer, epoch,best_miou):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_miou':best_miou,
    }
    ckpt_model_filename = "ckpt_{}_epoch_{}.pth".format(modelName,epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

def save_ckpt_bestmiou(ckpt_dir, model, modelName,optimizer, epoch,best_miou):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_miou':best_miou,
        'best_miou_epoch':epoch,
    }
    ckpt_model_filename = "Bestest_mIou_ckpt_{}_epoch_{}.pth".format(modelName,epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
        best_iou=checkpoint['best_miou']
        return epoch,model,best_iou
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)
