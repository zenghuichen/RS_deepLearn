import torch
from models.get_models import get_models
import argparse
from tensorboardX import SummaryWriter  # 记录文件
from utils.utils import get_optimizer,get_scheduler,save_ckpt,save_ckpt_bestmiou,load_ckpt
from utils.loginfomation import loginfomation
from utils.loss import *
from config.config import fcn_model_config
from datasetloader.get_datasetLoader import *
import time
from utils.metrics import runningScore
from utils.loss import get_lossfunction
from torchvision.utils import make_grid
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt


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
    E_train_loader_256,E_test_loader_256=get_dataset(datasetName=config_param["datasetName"],dataSize='E256',num_workers=config_param["num_work"]) 
    E_train_loader_512,E_test_loader_512=get_dataset(datasetName=config_param["datasetName"],dataSize='E512',num_workers=config_param["num_work"]) 
    # 创建模型
    model=get_models(config_param["modelName"],num_class=config_param['n_class'])
    optimizer=get_optimizer(model,config_param['optimizer'],lr=config_param['learn_rate'],momentum=0.9,weight_decay=0)
    scheduler=get_scheduler(optimizer,config_param['scheduler'],mode="min",patience=len(E_train_loader_256)) # min对应loss
    # 预加载模型
    startepoch=0
    best_iou=-1
    if  not  config_param['pre_model_path'] is None:
        startepoch,model,optimizer,best_iou=load_ckpt(model, optimizer, config_param['pre_model_path'],'cuda')
    # 创建加载数据集
    datasetLoader={'E256':[E_train_loader_256,E_test_loader_256],"E512":[E_train_loader_512,E_test_loader_512]}
    lossfunction=get_lossfunction(config_param)
    writer = SummaryWriter(config_param["writerpath"]) # 处理损失函数信息加载
    # 处理记录日文件
    logobject=loginfomation(config_param["modelName"],config_param["logpath"])
    ckpt_dir=config_param["checkpoint"]
    # 计算write
    return datasetLoader,model,optimizer,scheduler,lossfunction,writer,logobject,startepoch,ckpt_dir,writer,best_iou

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
        image,seg,label=sample['img'], sample['seg'], sample['label']
        images=image.cuda()
        # 模型训练
        optimizer.zero_grad()
        outputs = model(images)
        if muilt:
            loss=lossfunction(outputs,label.cuda())
        else:
            pass
        loss.backward()
        optimizer.step() 
        scheduler.step(loss)  # 调整学习率
        # 输出结果
        strlines.append("epoch:{},iter:{}/{},loss:{},learn_rate:{},itertime:{}".format(epoch,i,datalen,loss.detach().cpu().data, optimizer.param_groups[0]['lr'],time.time()-end))
        print(strlines[-1])
        # writer 加载数据
        if writer is None:
            global_step=global_step+1
            continue
        # 日常日志记录
        writer.add_scalar('train_loss', loss.data, global_step=global_step)
        writer.add_scalar('train_Learning_rate', optimizer.param_groups[0]['lr'], global_step=global_step)
        # 记录图片
        if i%image_step==0:
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('trian_image', grid_image, global_step)


            output_pre=torch.argmax(outputs, 1).reshape(-1,1,image.shape[2],image.shape[3])[:3].clone().cpu().data
            grid_image = make_grid(output_pre, 3, normalize=False,range=(0, 255))            
            writer.add_image('train_Predicted_label', grid_image, global_step)

            label_gt=label.reshape(-1,1,label.shape[1],label.shape[2])[:3].clone().cpu().data
            grid_image = make_grid(label_gt, 3, normalize=False, range=(0, 255))
            writer.add_image('train_Groundtruth_label', grid_image, global_step)


        global_step=global_step+1

    print("epoch:{},times:{}".format(epoch,time.time()-start))
    logobject.logTrainlog(strlines)
    print("本批次的运算时间:{}，平均时间:{}".format(end-start,(end-start)/datalen))

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
            image,seg,label=sample['img'], sample['seg'], sample['label']
            images=image.cuda()
            outputs = model(image.cuda())
            if muilt:
                loss=lossfunction(outputs,label.cuda())
            else:
                pass
            _, image_h, image_w = label.shape

            pred = F.interpolate(outputs, (image_h, image_w),
                                         mode='bilinear',
                                         align_corners=False)
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            gt = label.data.cpu().numpy()
            # 统计损失值
            running_metrics.update(gt, pred)
            strlines.append("epoch:{},iter:{}/{},loss:{},itertime:{}".format(epoch,i,datalen,loss.detach().cpu().data, time.time()-end))
            print(strlines[-1])
            # writer 加载数据
            if writer is None:
                global_step=global_step+1
                continue
            # 日常日志记录
            writer.add_scalar('test_loss', loss.data, global_step=global_step)
            ###writer.add_scalar('test_Learning_rate', scheduler.get_lr()[0], global_step=global_step)
            # 记录图片
            if i%image_step==0:
                grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('test_image', grid_image, global_step)
                output_pre=torch.argmax(outputs, 1).reshape(-1,1,outputs.shape[2],outputs.shape[3])[:3].clone().cpu().data
                grid_image = make_grid(output_pre, 3, normalize=False,range=(0, 255))  
                writer.add_image('test_Predicted_label', grid_image, global_step)
                label_gt=label.reshape(-1,1,label.shape[1],label.shape[2])[:3].clone().cpu().data
                grid_image = make_grid(label_gt, 3, normalize=False, range=(0, 255))
                writer.add_image('test_Groundtruth_label', grid_image, global_step)
            global_step=global_step+1
        print("epoch:{},times:{}".format(epoch,time.time()-start))
        logobject.logtestlog(strlines)
    return global_step,logobject,running_metrics

def save_model(ckpt_dir,epoch,model,modelName,optimizer,running_metrics,best_iou):
    scores_val, class_iou_val = running_metrics.get_scores()
    for k, v in scores_val.items():
        print(k+': %f' % v)
        # --------save best model --------
    if scores_val['Mean IoU'] >= best_iou:
        best_iou = scores_val['Mean IoU']
        print('Best model updated!')
        print(class_iou_val)
        best_model_stat = {'epoch': 0, 'scores_val': scores_val, 'class_iou_val': class_iou_val}
        #save_ckpt_bestmiou(ckpt_dir, model, modelName, optimizer, epoch, best_iou)
    #save_ckpt(ckpt_dir, model, modelName, optimizer, epoch,best_iou)
    running_metrics.reset()
    return scores_val,class_iou_val,best_iou,running_metrics

def mainTrain(config_param,isTest=True):
    datasetLoader,model,optimizer,scheduler,lossfunction,writer,logobject,startepoch,ckpt_dir,writer,best_iou=loadconfig(config_param)
    trainstep=0
    valstep=0
    teststep=0
    n_class=config_param['n_class']
    print("trainDataset=======> {}".format(config_param["datasetName"]))
    print("model sturct=======>")
    #print(model)
    print("start Train=======>")
    print("model {} =======>".format(config_param["modelName"]))
    model=model.cuda() # 使用显卡
    running_metrics = runningScore(n_class)
    for epoch in range(startepoch,config_param["maxepoch"]):
        if epoch<config_param['E512Step']:
            trainloader,testloader=datasetLoader['E256']
        else: # 切换数据源
            trainloader,testloader=datasetLoader['E512']
        # 开始考虑模型训练
        # 训练集
        trainstep,logobject,model,scheduler,optimizer=train(model,trainloader,epoch,n_class,optimizer,scheduler,lossfunction,logobject,muilt=True,global_step=trainstep,image_step=1,writer=writer)
        # 测试集
        teststep,logobject,running_metrics=test(model,testloader,running_metrics,epoch,lossfunction,logobject,muilt=True,global_step=teststep,image_step=1,writer=writer)
        # 输出结果
        scores_val,class_iou_val,best_iou,running_metrics=save_model(ckpt_dir,epoch,model,config_param["modelName"],optimizer,running_metrics,best_iou)
        logobject.logValInfo(epoch, scores_val, class_iou_val)

if __name__=="__main__":
    config_param=fcn_model_config
    mainTrain(config_param,isTest=True)