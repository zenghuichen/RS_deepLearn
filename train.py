import torch
from models.get_models import get_models
import argparse
from tensorboardX import SummaryWriter  # 记录文件
from utils.utils import get_optimizer,get_scheduler,save_ckpt,save_ckpt_bestmiou,load_ckpt
from utils.loginfomation import loginfomation
from utils.loss import *

from datasetloader.get_datasetLoader import *
import time
from utils.metrics import runningScore
from utils.loss import get_lossfunction
from torchvision.utils import make_grid
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
from config.config import fcn_model_config_RGB123,fcn_model_config_RGB124,fcn_model_config_RGB134,fcn_model_config_RGB234

def loadconfig(config_param):
    '''
    初始化数据结构
    '''
    if not config_param["writerpath"] is None and  not os.path.exists(config_param["writerpath"]):
        os.makedirs(config_param["writerpath"])

    if not config_param["logpath"] is None and  not os.path.exists(config_param["logpath"]):
        os.makedirs(config_param["logpath"])
                 
    if not config_param["checkpoint"] is None and  not os.path.exists(config_param["checkpoint"]):
        os.makedirs(config_param["checkpoint"])

    # 构建数据集
    E_train_loader_256,E_test_loader_256=get_dataset(datasetName=config_param["datasetName"],dataSize='E256',batch_size=config_param["batch_size"],num_workers=config_param["num_work"]) 
    E_train_loader_512,E_test_loader_512=get_dataset(datasetName=config_param["datasetName"],dataSize='E512',batch_size=config_param["batch_size"],num_workers=config_param["num_work"]) 
    # 创建模型
    model=get_models(config_param["modelName"],num_class=config_param['n_class'])
    optimizer=get_optimizer(model,config_param['optimizer'],lr=config_param['learn_rate'],momentum=0.9,weight_decay=0)
    scheduler=get_scheduler(optimizer,config_param['scheduler'],mode="min",patience=10*len(E_train_loader_256)) # min对应loss
    # 预加载模型
    startepoch=0
    best_iou=-1
    if  not  config_param['pre_model_path'] is None:
        startepoch,model,best_iou=load_ckpt(model, optimizer, config_param['pre_model_path'],'cuda')
    # 创建加载数据集
    datasetLoader={'E256':[E_train_loader_256,E_test_loader_256],"E512":[E_train_loader_512,E_test_loader_512]}
    lossfunction=get_lossfunction(config_param)
    writer = SummaryWriter(config_param["writerpath"]) # 处理损失函数信息加载
    # 处理记录日文件
    logobject=loginfomation(config_param["modelName"],config_param["logpath"])
    ckpt_dir=config_param["checkpoint"]
    # 计算write
    return datasetLoader,model,optimizer,scheduler,lossfunction,writer,logobject,startepoch,ckpt_dir,writer,best_iou


def DrawImage(img,pre,gt,logresult_dir,epoch=0,sig='train'):
    '''自动动画绘制图像 需要提前 plt.ion() 初始化绘制环境'''
    '''img,pre,gt 数据全部都为numpy'''
    plt.cla()
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('image')
    plt.subplot(1,3,2)
    plt.imshow(pre)
    plt.title('pred')
    plt.subplot(1,3,3)
    plt.imshow(gt[0,:,:])
    plt.title('gt')
    plt.pause(0.01) # 停留1s
    plt.savefig(os.path.join(logresult_dir,"sig_{}".format(epoch)))

def WriterSummary(writer,sig,loss,lr_rate,image,pred,gt,seg,step,image_step):
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
        pred_softmax=F.softmax(pred).max(1)[1][0,:,:].squeeze().detach().cpu().data
        pred_softmax=pred_softmax.reshape(1,pred_softmax.shape[0],pred_softmax.shape[1])
        pred_softmax=torch.cat([pred_softmax,pred_softmax,pred_softmax])
        gt_img=gt[0,0,:,:].detach().cpu().data
        gt_img=gt_img.reshape(1,gt_img.shape[0],gt_img.shape[1])
        gt_img=torch.cat([gt_img,gt_img,gt_img])

        seg_img=seg[0,:,:]
        seg_img=seg_img.reshape(1,seg_img.shape[0],seg_img.shape[1])
        seg_img=torch.cat([seg_img,seg_img,seg_img])
        writer_img=make_grid([img,pred_softmax,gt_img,seg_img],nrow=4)
        writer.add_image("{}_img".format(sig),writer_img,step)
    pass

def WriterAccurary(writer,scores_val,class_iou_val,epoch):
    for k, v in scores_val.items():
        writer.add_scalar("test_scores_{}".format(k),v,epoch)

    for k, v in class_iou_val.items():
        writer.add_scalar("test_class_{}".format(k),v,epoch)



def train(model,trainloader,epoch,n_classes,optimizer,scheduler,lossfunction,logobject,muilt=True,global_step=0,image_step=100,writer=None):
    '''
    训练模型
    '''
    datalen=len(trainloader)
    start=time.time()
    model.train()
    strlines=[]
    for i, sample in enumerate(trainloader):
        end=time.time()
        image,seg,labels,source_img=sample['img'], sample['seg'], sample['label'],sample["source_image"]
        images=image.cuda()
        # 模型训练
        optimizer.zero_grad()
        outputs = model(images)
        if muilt:
            loss=lossfunction(outputs,labels.cuda())
            #loss=lossfunction(F.log_softmax(outputs),label.cuda())
        else:
            pass
        loss.backward()
        optimizer.step() 
        scheduler.step(loss)  # 调整学习率
        # 输出结果
        strlines.append("train epoch:{},iter:{}/{},loss:{},learn_rate:{},itertime:{}".format(epoch,i,datalen,loss.detach().cpu().data, optimizer.param_groups[0]['lr'],time.time()-end))
        print(strlines[-1])
        # writer 加载数据
        if writer is None:
            global_step=global_step+1
            continue
        # 日常日志记录
        WriterSummary(writer,'train',loss,optimizer.param_groups[0]['lr'],source_img,outputs,labels,seg,global_step,image_step)

        global_step=global_step+1

    #print("epoch:{},times:{}".format(epoch,time.time()-start))
    logobject.logTrainlog(strlines)
    #print("本批次的运算时间:{}，平均时间:{}".format(end-start,(end-start)/datalen))

    return global_step,logobject,model,scheduler,optimizer

def test(model,testloader,running_metrics,epoch,lossfunction,logobject,muilt=True,global_step=0,image_step=100,writer=None):
    '''
    测试数据集
    '''
    datalen=len(testloader)
    start=time.time()
    model.eval()
    strlines=[]
    with torch.no_grad():
        for i, sample in enumerate(testloader):
            end=time.time()
            image,seg,label,source_img=sample['img'], sample['seg'], sample['label'],sample["source_image"]
            images=image.cuda()
            outputs = model(image.cuda())
            if muilt:
                loss=lossfunction(outputs,label.cuda())
            else:
                pass
            _, _,image_h, image_w = label.shape

            pred=F.softmax(outputs).cpu() # 因为原始模型中结尾 没有softmax

            #pred = torch.argmax(pred, dim=1).cpu()
            gt = label.data.cpu()
            # 统计损失值
            running_metrics.update(gt.numpy(), pred.numpy()>0.5)
            strlines.append("test epoch:{},iter:{}/{},loss:{},itertime:{}".format(epoch,i,datalen,loss.detach().cpu().data, time.time()-end))
            print(strlines[-1])
            # writer 加载数据
            if writer is None:
                global_step=global_step+1
                continue
            # 日常日志记录
            WriterSummary(writer,'test',loss,0.01,source_img,outputs,label,seg,global_step,image_step=image_step)

            global_step=global_step+1
        print("epoch:{},times:{}".format(epoch,time.time()-start))
        logobject.logtestlog(strlines)
    return global_step,logobject,running_metrics

def save_model(ckpt_dir,epoch,model,modelName,optimizer,running_metrics,best_iou,datasetName,issaveBestIOU=False):
    scores_val, class_iou_val = running_metrics.get_scores()
    for k, v in scores_val.items():
        print(k+': %f' % v)

    for k, v in class_iou_val.items():
        print(str(k)+': %f' % v)    
        
        # --------save best model --------
    if scores_val['Mean IoU'] > best_iou and issaveBestIOU:
        best_iou = scores_val['Mean IoU']
        print('Best model updated!')
        print(class_iou_val)
        best_model_stat = {'epoch': 0, 'scores_val': scores_val, 'class_iou_val': class_iou_val}
        save_ckpt_bestmiou(ckpt_dir, model, modelName, optimizer, epoch, best_iou,datasetName)

    running_metrics.reset()
    return scores_val,class_iou_val,best_iou,running_metrics

def mainTrain(config_param,isTest=True):
    datasetLoader,model,optimizer,scheduler,lossfunction,writer,logobject,startepoch,ckpt_dir,writer,best_iou=loadconfig(config_param)
    trainstep=0
    valstep=0
    teststep=0
    n_class=config_param['n_class']
    print("trainDataset=======> {}".format(config_param["datasetName"]))
    print("model {} =======>".format(config_param["modelName"]))
    model=model.cuda() # 使用显卡
    running_metrics = runningScore(n_class)


    # 自动绘制图像
    plt.figure(figsize=(13,3))
    plt.ion()

    for epoch in range(startepoch,config_param["maxepoch"]):
        if epoch<config_param['E512Step']:
            trainloader,testloader=datasetLoader['E256']
        else: # 切换数据源
            trainloader,testloader=datasetLoader['E512']
        # 开始考虑模型训练
        # 训练集
        trainstep,logobject,model,scheduler,optimizer=train(model,trainloader,epoch,n_class,optimizer,scheduler,lossfunction,logobject,muilt=True,global_step=trainstep,image_step=100,writer=writer)
        # 测试集
        teststep,logobject,running_metrics=test(model,testloader,running_metrics,epoch,lossfunction,logobject,muilt=True,global_step=teststep,image_step=100,writer=writer)
        # 输出结果
        issavebestModel=epoch>=config_param['E512Step']
        scores_val,class_iou_val,best_iou,running_metrics=save_model(ckpt_dir,epoch,model,config_param["modelName"],optimizer,running_metrics,best_iou,config_param['datasetName'],issaveBestIOU=issavebestModel)
        WriterAccurary(writer,scores_val,class_iou_val,epoch)
        logobject.logValInfo(epoch, scores_val, class_iou_val)
        if epoch%10==0:
            save_ckpt(ckpt_dir, model, config_param["modelName"], optimizer, epoch,best_iou,config_param['datasetName'])
if __name__=="__main__":
    '''FCN 模型训练 RGB'''
    config_param=fcn_model_config_RGB123 # RGB123
    mainTrain(config_param,isTest=True)
    config_param=fcn_model_config_RGB124 # RGB124
    mainTrain(config_param,isTest=True)
    config_param=fcn_model_config_RGB134 # RGB134
    mainTrain(config_param,isTest=True)
    config_param=fcn_model_config_RGB234 # RGB234
    mainTrain(config_param,isTest=True)
