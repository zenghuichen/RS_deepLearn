import torch
from models.get_models import get_models
import argparse
from tensorboardX import SummaryWriter  # 记录文件
from utils.utils import get_optimizer,get_scheduler,save_ckpt,save_ckpt_bestmiou,load_ckpt
from utils.loginfomation import loginfomation,WriterAccurary,WriterSummary
from utils.loss import *

from datasetloader.get_datasetLoader import *
import time
from utils.metrics import runningScore
from utils.loss import get_lossfunction
from torchvision.utils import make_grid
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
from config.config_fcn import fcn_model_config_RGB123,fcn_model_config_RGB124,fcn_model_config_RGB134,fcn_model_config_RGB234
from config.config_unet import unet_model_config_RGB123,unet_model_config_RGB124,unet_model_config_RGB134,unet_model_config_RGB234
from config.config_segnet import segnet_model_config_RGB123,segnet_model_config_RGB124,segnet_model_config_RGB134,segnet_model_config_RGB234
from config.config_VI import segnet_model_config_VI,unet_model_config_VI,fcn_model_config_VI
import sys
import threading
from time import sleep

class DatasetLoader_Thread(threading.Thread):
    def __init__(self,func,args,name=""):
        threading.Thread.__init__(self)
        self.func=func
        self.name=name
        self.args=args

    def run(self):
        self.res=self.func(*self.args)

    def getResult(self):
        return self.res

def generator_datasetloader_iter(config_param,dataSize):
    if sys.platform=="win32":
        E_train_loader, E_test_loader = get_dataset(config_param,dataSize=dataSize) 
        trainloader_iter = data_prefetcher(E_train_loader)
        testloader_iter = data_prefetcher(E_test_loader)
        return trainloader_iter,testloader_iter,len(E_train_loader),len(E_test_loader)
    elif sys.platform=='linux':
        E_train_loader, E_test_loader = get_dataset(config_param,dataSize=dataSize)
        return  E_train_loader,E_test_loader,len(E_train_loader),len(E_test_loader)

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

    # 创建模型            
    model=get_models(config_param["modelName"],num_class=config_param['n_class'],channels_num=config_param['channels'])
    optimizer=get_optimizer(model,config_param['optimizer'],lr=config_param['learn_rate'],momentum=0.9,weight_decay=0)
    
    # 预加载模型
    startepoch=0
    best_iou=-1
    if not config_param['pre_model_path'] is None:
        startepoch,model,best_iou=load_ckpt(model, optimizer, config_param['pre_model_path'],'cuda')
    # 创建加载数据集
    lossfunction=get_lossfunction(config_param)
    writer = SummaryWriter(config_param["writerpath"]) # 处理损失函数信息加载
    # 处理记录日文件
    logobject=loginfomation(config_param["modelName"],config_param["logpath"])
    ckpt_dir=config_param["checkpoint"]
    # 计算write
    return model,optimizer,lossfunction,writer,logobject,startepoch,ckpt_dir,writer,best_iou


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

def train_main_win32(config_param,muilt=True):
    '''构建一个迭代池 pool '''

    model, optimizer, lossfunction, writer, logobject, startepoch, ckpt_dir, writer, best_iou = loadconfig(config_param)
    n_class=config_param['n_class']
    print("trainDataset=======> {}".format(config_param["datasetName"]))
    print("model {} =======>".format(config_param["modelName"]))
    model=model.cuda() # 使用显卡
    running_metrics = runningScore(n_class)

    ''' 训练时 '''
    global_step_train = 0
    global_step_test = 0
    best_iou=-1
    datasetname= config_param['datasetName']
    
    
    trainloader_iter, testloader_iter, train_len, test_len=generator_datasetloader_iter(config_param,'E256')
    trainloader_ls=[trainloader_iter]
    testloader_ls=[testloader_iter]

    scheduler=get_scheduler(optimizer,config_param['scheduler'],mode="min",patience=2*train_len) # min对应loss

    for epoch in range(startepoch,config_param["maxepoch"]):
        #trainloader_iter, testloader_iter, train_len, test_len = generator_datasetloader_iter(config_param)
        trainloader_iter=trainloader_ls.pop()
        testloader_iter=testloader_ls.pop()
        '''多线程构建数据集加载对象'''
        if sys.platform=="win32":
            if epoch>config_param['E512Step']-1:
                dataserLoader_iter_th=DatasetLoader_Thread(generator_datasetloader_iter,(config_param,'E512',))
                dataserLoader_iter_th.start()
            else:
                dataserLoader_iter_th=DatasetLoader_Thread(generator_datasetloader_iter,(config_param,'E256',))
                dataserLoader_iter_th.start()

        '''模型训练阶段'''
        start = time.time()
        model.train()
        strlines = []
        datalen = train_len
        sample=trainloader_iter.next()
        d_i=global_step_train
        while sample is not None:
            i=global_step_train-d_i
            end=time.time()
            image,labels,source_img,seg=sample['img'], sample['label'],sample["source_image"],sample['seg']
            images=image.cuda()
            # 模型训练
            optimizer.zero_grad()
            outputs = model(images)
            if muilt:
                loss=lossfunction(outputs,labels.cuda())
                if torch.isnan(loss):
                    print("because loss is nan,the process of train must stop,and the program also exit.")
                    return "nan"
                #loss=lossfunction(F.log_softmax(outputs),label.cuda())
            else:
                pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss)  # 调整学习率
            # 输出结果
            strlines.append("{} train epoch:{},iter:{}/{},loss:{},learn_rate:{},itertime:{}".format(datasetname,epoch,i,datalen,loss.detach().cpu().data, optimizer.param_groups[0]['lr'],time.time()-end))
            print("\r{}".format(strlines[-1]),end="")
            # writer 加载数据
            if not writer is None:
                #pass
                WriterSummary(writer, 'train', loss, optimizer.param_groups[0]['lr'], source_img, outputs, labels,seg,global_step_train, image_step=200)
            # 日常日志记录
            global_step_train=global_step_train+1
            sample = trainloader_iter.next()

        print("\nepoch:{},times:{}".format(epoch,time.time()-start))
        logobject.logTrainlog(strlines)

        ''' 模型验证阶段'''
        datalen = test_len
        start = time.time()
        model.eval()
        strlines = []

        with torch.no_grad():
            d_i=global_step_test
            sample=testloader_iter.next()
            while sample is not None:
                i=global_step_test-d_i
                end = time.time()
                image,label,source_img,seg=sample['img'],  sample['label'],sample["source_image"],sample['seg']
                images = image.cuda()
                outputs = model(image.cuda())
                if muilt:
                    loss = lossfunction(outputs, label.cuda())
                else:
                    pass
                _, _, image_h, image_w = label.shape

                pred = F.softmax(outputs).cpu()  # 因为原始模型中结尾 没有softmax

                # pred = torch.argmax(pred, dim=1).cpu()
                gt = label.data.cpu()
                # 统计损失值
                running_metrics.update(gt.numpy(), pred.numpy() > 0.5)
                strlines.append(
                    "{}  test epoch:{},iter:{}/{},loss:{},itertime:{}".format(datasetname, epoch, i, datalen,
                                                                              loss.detach().cpu().data,
                                                                              time.time() - end))
                print("\r{}".format(strlines[-1]), end="")
                # writer 加载数据
                if not writer is None:
                    WriterSummary(writer, 'test', loss, 0.01, source_img, outputs, label, seg,global_step_test,image_step=60)
                # 日常日志记录
                global_step_test = global_step_test + 1
                sample = testloader_iter.next()
            print("\nepoch:{},times:{}".format(epoch, time.time() - start))
            logobject.logtestlog(strlines)

        '''模型度量阶段'''
        scores_val,class_iou_val,best_iou,running_metrics=save_model(ckpt_dir,epoch,model,config_param["modelName"],optimizer,running_metrics,best_iou,config_param['datasetName'])
        WriterAccurary(writer,scores_val,class_iou_val,epoch)
        ''' 记录模型'''
        logobject.logValInfo(epoch, scores_val, class_iou_val)
        if epoch % 10 == 0:
            save_ckpt(ckpt_dir, model, config_param["modelName"], optimizer, epoch, best_iou, config_param['datasetName'])

        if sys.platform =="linux":
            if epoch>config_param['E512Step']-1:
                trainloader_iter, testloader_iter, train_len, test_len=generator_datasetloader_iter(config_param,'E512')
            else:
                trainloader_iter, testloader_iter, train_len, test_len=generator_datasetloader_iter(config_param,'E256')
            trainloader_ls=[trainloader_iter]
            testloader_ls=[testloader_iter]        
        elif sys.platform=="win32":
            if not dataserLoader_iter_th is None:
                dataserLoader_iter_th.join()
                trainloader_iter, testloader_iter, train_len, test_len=dataserLoader_iter_th.getResult()
                trainloader_ls.append(trainloader_iter)
                testloader_ls.append(testloader_iter)


    writer.close()
    print("===> {} over".format(config_param["datasetName"]))


def train_main_ubuntu(config_param,muilt=True):

    model, optimizer, lossfunction, writer, logobject, startepoch, ckpt_dir, writer, best_iou = loadconfig(config_param)
    n_class=config_param['n_class']
    print("trainDataset=======> {}".format(config_param["datasetName"]))
    print("model {} =======>".format(config_param["modelName"]))
    model=model.cuda() # 使用显卡
    running_metrics = runningScore(n_class)

    ''' 训练时 '''
    global_step_train = 0
    global_step_test = 0
    best_iou=-1
    datasetname= config_param['datasetName']
    
    trainloader_256, testloader_256, train_len_256, test_len_256=generator_datasetloader_iter(config_param,'E256')
    trainloader_512, testloader_512, train_len_512, test_len_512=generator_datasetloader_iter(config_param,'E512')

    scheduler=get_scheduler(optimizer,config_param['scheduler'],mode="min",patience=2*train_len_256) # min对应loss

    for epoch in range(startepoch,config_param["maxepoch"]):

        '''模型训练阶段'''
        start = time.time()
        model.train()
        strlines = []
        '''数据集加载'''
        if epoch>config_param['E512Step']:
            train_loader=trainloader_512
            test_loader=testloader_512
            datalen = train_len_512
        else:
            train_loader=trainloader_256
            test_loader=testloader_256
            datalen = train_len_256

        for i,sample in enumerate(train_loader):
            end=time.time()
            image,labels,source_img,seg=sample['img'], sample['label'],sample["source_image"],sample['seg']
            images=image.cuda()
            # 模型训练
            optimizer.zero_grad()
            outputs = model(images)
            if muilt:
                loss=lossfunction(outputs,labels.cuda())
                if torch.isnan(loss):
                    print("because loss is nan,the process of train must stop,and the program also exit.")
                    return "nan"
            else:
                pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss)  # 调整学习率
            # 输出结果
            strlines.append("{} train epoch:{},iter:{}/{},loss:{},learn_rate:{},itertime:{}".format(datasetname,epoch,i,datalen,loss.detach().cpu().data, optimizer.param_groups[0]['lr'],time.time()-end))
            print("\r{}".format(strlines[-1]),end="")
            # writer 加载数据
            if not writer is None:
                #pass
                WriterSummary(writer, 'train', loss, optimizer.param_groups[0]['lr'], source_img, outputs, labels,seg,global_step_train, image_step=200)
            # 日常日志记录
            global_step_train=global_step_train+1

        print("\nepoch:{},times:{}".format(epoch,time.time()-start))
        logobject.logTrainlog(strlines)

        ''' 模型验证阶段'''
        start = time.time()
        model.eval()
        strlines = []

        with torch.no_grad():
            for i,sample in enumerate(test_loader):
                end = time.time()
                image,label,source_img,seg=sample['img'],  sample['label'],sample["source_image"],sample['seg']
                images = image.cuda()
                outputs = model(image.cuda())
                if muilt:
                    loss = lossfunction(outputs, label.cuda())
                else:
                    pass
                _, _, image_h, image_w = label.shape

                pred = F.softmax(outputs).cpu()  # 因为原始模型中结尾 没有softmax

                # pred = torch.argmax(pred, dim=1).cpu()
                gt = label.data.cpu()
                # 统计损失值
                running_metrics.update(gt.numpy(), pred.numpy() > 0.5)
                strlines.append(
                    "{}  test epoch:{},iter:{}/{},loss:{},itertime:{}".format(datasetname, epoch, i, datalen,
                                                                              loss.detach().cpu().data,
                                                                              time.time() - end))
                print("\r{}".format(strlines[-1]), end="")
                # writer 加载数据
                if not writer is None:
                    WriterSummary(writer, 'test', loss, 0.01, source_img, outputs, label, seg,global_step_test,image_step=60)
                # 日常日志记录
                global_step_test = global_step_test + 1
            print("\nepoch:{},times:{}".format(epoch, time.time() - start))
            logobject.logtestlog(strlines)

        '''模型度量阶段'''
        scores_val,class_iou_val,best_iou,running_metrics=save_model(ckpt_dir,epoch,model,config_param["modelName"],optimizer,running_metrics,best_iou,config_param['datasetName'])
        WriterAccurary(writer,scores_val,class_iou_val,epoch)
        ''' 记录模型'''
        logobject.logValInfo(epoch, scores_val, class_iou_val)
        if epoch % 10 == 0:
            save_ckpt(ckpt_dir, model, config_param["modelName"], optimizer, epoch, best_iou, config_param['datasetName'])

    writer.close()
    print("===> {} over".format(config_param["datasetName"]))


def train_main(config_param,muilt=True):
    if sys.platform=="win32":
        train_main_win32(config_param,muilt=muilt)
    elif sys.platform=="linux":
        train_main_ubuntu(config_param,muilt=muilt)


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

if __name__=="__main__":
    # 自动绘制图像
    plt.figure(figsize=(6,2))
    plt.ion()
    #config_param=fcn_model_config_RGB123 # RGB123
    #mainTrain(config_param,isTest=True)
    errmodels=[]
    # 修改为训练代码
    config_params=[fcn_model_config_RGB123,fcn_model_config_RGB124,fcn_model_config_RGB134,fcn_model_config_RGB234,
                #unet_model_config_RGB123,unet_model_config_RGB124,unet_model_config_RGB134,unet_model_config_RGB234,
                #segnet_model_config_RGB123,segnet_model_config_RGB124,segnet_model_config_RGB134,segnet_model_config_RGB234,
                #unet_model_config_VI,
                #fcn_model_config_VI,
                #segnet_model_config_VI, # 归一化指数
                 ]
    for config_param in config_params:
        print("当前工作路径为：{},部分参数修正为".format(os.getcwd()))
        # 参数路径修复
        for i in ["writerpath","logpath",'checkpoint']:
            writerpath=config_param[i]
            config_param[i]=os.path.join(os.getcwd(),config_param[i])
            print("{}: {} ===> {}".format(i,writerpath,config_param[i]))
        trrainsig=train_main(config_param,muilt=True)
        if trrainsig =="nan":
            errmodels.append(config_param)
            print("the loss of train is nan !!!! model Name:{} , Dataset Name：{}".format(config_param["modelName"],config_param['datasetName']))
    plt.close()
    print("出现问题的模型，需要单独训练，并调整相应的loss值缩放系数")
    for config_param in errmodels:
        print("the loss of train is nan !!!! model Name:{} , Dataset Name：{}".format(config_param["modelName"],config_param['datasetName']))
    
