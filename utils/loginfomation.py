import os
import numpy as np
import pandas as pds
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import make_grid

class loginfomation(object):
    def __init__(self,model_name,rootdir):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        self.model_name=model_name
        self.csvpath=os.path.join(rootdir,"val_log_{}.csv".format(model_name))
        self.logtrainpath=os.path.join(rootdir,"trian_log_{}.txt".format(model_name))
        self.logvalpath=os.path.join(rootdir,"val_log_{}.txt".format(model_name))
        self.logtestpath=os.path.join(rootdir,"test_log_{}.txt".format(model_name))
        
        self.loginfo={"modelname":[],'epoch':[]}

    def logValInfo(self,epoch,scores_val,class_iou_val):
        self.loginfo["epoch"].append(epoch)
        self.loginfo['modelname'].append(self.model_name)
        for k, v in scores_val.items():
            if k not in self.loginfo:
                self.loginfo[k]=[]
            self.loginfo[k].append(v)
        for k, v in class_iou_val.items():
            if k not in self.loginfo:
                self.loginfo[k]=[]
            self.loginfo[k].append(v) 
        # 输出至普通文件
        csvobj=pds.DataFrame(self.loginfo)
        csvobj.to_csv(self.csvpath,encoding='utf-8')

    def logTrainlog(self,stringline):
        with open(self.logtrainpath,'a',encoding='utf-8') as fp:
            for lines in stringline:
                fp.write("{}\n".format(lines))
    
    def logvallog(self,stringline):
        with open(self.logvalpath,'a',encoding='utf-8') as fp:
            for lines in stringline:
                fp.write("{}\n".format(lines))
    
    def logtestlog(self,stringline):
        with open(self.logtestpath,'a',encoding='utf-8') as fp:
            for lines in stringline:
                fp.write("{}\n".format(lines))    

def DrawImage(img,pre,gt,pre_softmax,seg,logresult_dir=None,epoch=0,sig='train'):
    '''自动动画绘制图像 需要提前 plt.ion() 初始化绘制环境'''
    '''img,pre,gt 数据全部都为numpy'''
    '''这里选择重新调整样本集的展示，将训练结果与实际结果合并展示成为一张图，并考虑写入训练日志中'''
    img=np.transpose(img,(1,2,0)).astype(np.int)
    pre=np.transpose(pre,(1,2,0)).astype(np.float)[:,:,0].reshape(pre.shape[1],pre.shape[2],1)
    gt=np.transpose(gt,(1,2,0)).astype(np.float)[:,:,0].reshape(gt.shape[1],gt.shape[2],1)
    seg=np.transpose(seg,(1,2,0)).astype(np.float)[:,:,0].reshape(seg.shape[1],seg.shape[2],1)
    pre_softmax=pre_softmax
    pre_gt=np.concatenate([pre,gt,seg],axis=2)
    plt.cla()
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('image_{}'.format(sig))
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(pre_gt)
    plt.title('pred_{}'.format(sig))
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pre_softmax)
    plt.title('pre_softmax_{}'.format(sig))
    plt.axis('off')

    plt.pause(0.01) # 停留1s
    if not logresult_dir is None:
        plt.savefig(os.path.join(logresult_dir,"sig_{}".format(epoch)))


def WriterSummary(writer,sig,loss,lr_rate,image,pred,gt,seg,step,image_step,isShow=True,logresult_dir=None,):
    '''
    记录当前数据的情况，并记录训练中的相关信息
    '''
    writer.add_scalar("{}_loss".format(sig),loss.data,step)
    if sig=="train":
        writer.add_scalar("{}_lr".format(sig),lr_rate,step)
    _,_,h,w=image.shape
    if step%image_step==0:
        # 保存训练数据
        img=image[0,:3,:,:].clone().cpu().data.reshape(3,h,w)
        pred_softmax=F.softmax(pred)
        pred_softmax_1=torch.argmax(pred_softmax,dim=1)
        pred_softmax_1=pred_softmax_1[0,:,:].squeeze().detach().cpu().data
        pred_softmax_1=pred_softmax_1.reshape(1,pred_softmax_1.shape[0],pred_softmax_1.shape[1])
        pred_softmax_1=torch.cat([pred_softmax_1,pred_softmax_1,pred_softmax_1])
        pred_softmax_1=pred_softmax_1

        gt_img=gt[0,0,:,:].detach().cpu().data
        gt_img=gt_img
        gt_img=gt_img.reshape(1,gt_img.shape[0],gt_img.shape[1])
        gt_img=torch.cat([gt_img,gt_img,gt_img])

        seg_img=seg[0,:,:]
        seg_img=seg_img.reshape(1,seg_img.shape[0],seg_img.shape[1])
        seg_img=torch.cat([seg_img,seg_img,seg_img])
        writer_img=make_grid([img,pred_softmax_1,gt_img,seg_img],nrow=4)

        if isShow:
            DrawImage(img.cpu().numpy(),
                        pred_softmax_1.detach().cpu().numpy(),
                        gt_img.cpu().numpy(),
                        pred_softmax.detach()[0,1,:,:].cpu().numpy(),
                        seg_img.cpu().numpy(),
                        epoch=image_step,
                        logresult_dir=logresult_dir,
                        sig=sig)

        writer.add_image("{}_img".format(sig),writer_img,step)
    pass

def WriterAccurary(writer,scores_val,class_iou_val,epoch):
    for k, v in scores_val.items():
        writer.add_scalar("test_scores_{}".format(k),v,epoch)

    for k, v in class_iou_val.items():
        writer.add_scalar("test_class_{}".format(k),v,epoch)

