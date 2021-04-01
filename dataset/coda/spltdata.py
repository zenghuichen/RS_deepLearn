'''
tiff 文件读写
https://blog.csdn.net/t46414704152abc/article/details/77482747
'''
from osgeo import gdal
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pds 
import json
import shutil


def ReadRSImg(imgpath):
    dataset=gdal.Open(imgpath)
    im_width=dataset.RasterXSize # 列数
    im_height=dataset.RasterYSize # 行数
    im_bands=dataset.RasterCount # 波段数
    im_geotrans=dataset.GetGeoTransform() # 仿射矩阵
    im_proj=dataset.GetProjection() # 地图投影信息
    # 数据文件
    data=np.zeros((im_height,im_width,im_bands))
    for i in range(im_bands):
        sigBandData=dataset.GetRasterBand(i+1)
        sigBandData_arr=sigBandData.ReadAsArray(0,0,im_width,im_height)
        data[:,:,i]=sigBandData_arr
    return data,im_geotrans,im_proj,dataset

def writeTIFF(imgdata,path,im_geotrans,im_proj):
    datatype=gdal.GDT_Float32 # 定义浮点数据
    if len(imgdata.shape)==2:
        im_h,im_w=imgdata.shape
        im_b=1
    elif len(imgdata.shape)==3:
        im_h,im_w,im_b=imgdata.shape
    
    # 创建文件
    driver=gdal.GetDriverByName("GTiff")
    dataset=driver.Create(path,im_w,im_h,im_b,datatype)
    if dataset!=None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    if im_b==1:
        dataset.GetRasterBand(1).WriteArray(imgdata)
    else:
        for i in range(im_b):
            dataset.GetRasterBand(i+1).WriteArray(imgdata[:,:,i])
    del dataset
    return os.path.exists(path)


def GeneratorSample(mask,dataset_mask,gf2data,imgsize,dw=0,dh=0):
    '''
    生成指定大小的 样本
    mask:掩膜板
    dataset_mask:掩摸般
    gfdata: 高分数据
    imgsize:样本大小
    dx: 增量dx
    dy: 增量dy
    '''
    h,w=mask.shape # mask 的形状
    start_w=dw # 开始x
    start_h=dh # 开始y
    ids=1
    result=[]
    for tw in range(start_w,w,imgsize):
        for th in range(start_h,h,imgsize):
            if th+imgsize>h or tw+imgsize>w:
                continue
            temp=mask[th:th+imgsize,tw:tw+imgsize]
            
            if np.sum(temp)==0:
                temp=temp*0 # 0 表示 不参与训练与测试，
                temp=temp+ids*10+0
                dataset_mask[th:th+imgsize,tw:tw+imgsize]=temp# 末尾为0
            else: # 此时存在对一个对应的label对象，可以用作训练集
                if random.random() <0.8:
                    temp=temp*0 # 0 表示 不参与训练与测试，
                    temp=temp+ids*10+1
                    dataset_mask[th:th+imgsize,tw:tw+imgsize]=temp  # 训练集标注为1
                    # 裁剪mask
                    result.append({
                        'ids':ids,
                        'mask':mask[th:th+imgsize,tw:tw+imgsize],
                        'data':gf2data[th:th+imgsize,tw:tw+imgsize],
                        'size':[th,tw,th+imgsize,tw+imgsize],
                        'sig':1 # 训练
                    })
                else: # 测试集标注为2
                    temp=temp*0 # 0 表示 不参与训练与测试，
                    temp=temp+ids*10+2
                    dataset_mask[th:th+imgsize,tw:tw+imgsize]=temp
                    result.append({
                        'ids':ids,
                        'mask':mask[th:th+imgsize,tw:tw+imgsize],
                        'data':gf2data[th:th+imgsize,tw:tw+imgsize],
                        'size':[th,tw,th+imgsize,tw+imgsize],
                        'sig':2 # 测试
                    })
            ids=ids+1
    return result,dataset_mask

def getEdge(mask):
    dx,dy=np.gradient(mask)
    dxy=np.abs(dx)+np.abs(dy)
    dxy=np.abs(dxy)>0
    dxy=dxy*1
    return dxy.astype(np.uint8)

def splitchildData(tname,tsig,tmask,tdata,imgsize=256):
    # 切分子 256 数据集
    w,h=tmask.shape
    temp_256=[]
    t_id=1
    for tw in range(0,w,imgsize):
        for th in range(0,h,imgsize):
            temp_256.append({
                'ids':"{}_{}".format(tname,t_id),
                'mask':tmask[th:th+imgsize,tw:tw+imgsize],
                'data':tdata[th:th+imgsize,tw:tw+imgsize],
                'sig':tsig
            })
            t_id=t_id+1
    return temp_256    

def SaveData(savedir,savetemp,minvalue,maxvalue):
    '''
    保存数据集
    '''
    tname,tmask,tdata,tsig=savetemp['ids'],savetemp['mask'],savetemp['data'],savetemp['sig']
    tedge=getEdge(tmask)
    # 保存数据
    SaveDataPath(tname,tmask,tdata,tsig,tedge,os.path.join(savedir,"ALLBands"),minvalue,maxvalue,isall=True)
    # RGB运算变换

    tdata=(tdata-minvalue)/(maxvalue-minvalue)
    tdata=tdata*255
    tdata=np.clip(tdata,0,255)
    tdata=tdata.astype(np.uint8) # 

    SaveDataPath(tname,tmask,tdata[:,:,[0,1,2]],tsig,tedge,os.path.join(savedir,"RGB123"),minvalue,maxvalue,isall=False)
    SaveDataPath(tname,tmask,tdata[:,:,[0,1,3]],tsig,tedge,os.path.join(savedir,"RGB124"),minvalue,maxvalue,isall=False)
    SaveDataPath(tname,tmask,tdata[:,:,[0,2,3]],tsig,tedge,os.path.join(savedir,"RGB134"),minvalue,maxvalue,isall=False)
    SaveDataPath(tname,tmask,tdata[:,:,[1,2,3]],tsig,tedge,os.path.join(savedir,"RGB234"),minvalue,maxvalue,isall=False)
    return True

def SaveDataPath(tname,tmask,tdata,tsig,tedge,savedir,minvalue,maxvalue,isall=False):

    if tsig ==1:
        savedir_t=os.path.join(savedir,'train')
    elif tsig==2:
        savedir_t=os.path.join(savedir,'test')
    else:
        return False

    # 创建对应的文件地址
    if not os.path.exists(os.path.join(savedir_t,'img')):
        os.makedirs(os.path.join(savedir_t,'img'))
    if not os.path.exists(os.path.join(savedir_t,'seg')):
        os.makedirs(os.path.join(savedir_t,'seg'))
    if not os.path.exists(os.path.join(savedir_t,'label')):
        os.makedirs(os.path.join(savedir_t,'label'))


    # 开始保存数据
    np.save(os.path.join(savedir_t,'img',"{}".format(tname)),tdata)
    np.save(os.path.join(savedir_t,'seg',"{}".format(tname)),tedge)
    np.save(os.path.join(savedir_t,'label',"{}".format(tname)),tmask)
    print(os.path.join(savedir_t,'img',"{}".format(tname)))
    return True

def saveSpliteResult(savedir,result,minvalue,maxvalue):
    '''
    savedir : 保存的文件地址
    result : 已经存在的结果
    '''

    E512_path=os.path.join(savedir,'E512')
    E256_path=os.path.join(savedir,'E256')

    # # 保存图像
    for t in result:
        tname,tmask,tdata,tsig=t['ids'],t['mask'],t['data'],t['sig']
        E256_temp=splitchildData(tname,tsig,tmask,tdata,imgsize=256) # 子类别
        for tt in E256_temp:
            SaveData(os.path.join(savedir,"E256"),tt,minvalues,maxvalues)
        SaveData(os.path.join(savedir,'E512'),t,minvalues,maxvalues)
    pass

def copyDatasetForTest(source_rootdir,target_rootdir,k_num=16):
    # 确定需要复制的代码文件
    for k in ['test','train']:
        # 确定需要复制的代码
        file_name_ls=os.listdir(os.path.join(source_rootdir,k,'img'))[:k_num]
        for kk in ['img','label','seg']:
            if not os.path.exists(os.path.join( target_rootdir,k,kk)):
                os.makedirs(os.path.join(target_rootdir,k,kk))
        # 复制文件
            for kk_name in file_name_ls:
                shutil.copy(os.path.join(source_rootdir,k,kk,kk_name),os.path.join(target_rootdir,k,kk,kk_name))
    

'''
train 1 epoch ->model in train
test(val) 1 epoch --> model in eval for acc  if better acc save model 
512x512  using random crop 256x256x256x256x4x2
'''

if __name__=='__main__':
    # 确定影像数据集的位置
    gf2_nanchang_path='/media/gis/databackup/ayc/modellist/dataset/tifData/bandhechengTIF.tif' 
    train_gf2_data_path='/media/gis/databackup/ayc/modellist/dataset/tifData/train_Mask_bandhechengTIF.tif' 
    traindataset="/media/gis/databackup/ayc/modellist/dataset/nanchang"
    data,im_geotrans,im_proj,dataset=ReadRSImg(gf2_nanchang_path)
    # 分离mask和data
    mask=data[:,:,0]
    gf2data=data[:,:,1:]
    # 数据的基本信息

    # 处理mask
    mask=mask==1 # 滤除其他类的情况
    mask=mask.astype(np.uint8) 
    # 准备生成训练mask ，确定训练的数据集位置与测试集位置
    train_test_mask=np.copy(mask)# 作为标记训练集与测试集
    train_test_mask=train_test_mask.astype(np.int32) # 原始值域大小为0-255
    # 首先切分512x512大小的数据集，并去出其中的不参与训练的位置
    E512,train_test_mask=GeneratorSample(mask,train_test_mask,gf2data,512) # 获得
    # 保存切分完全的训练集结果
    #data=np.concatenate([train_test_mask,data],axis=2) # 数据连接并保存
    print(train_test_mask.shape)
    print("min",np.min(train_test_mask))
    print("max",np.max(train_test_mask))
    writeTIFF(train_test_mask,train_gf2_data_path,im_geotrans,im_proj) # 保存对应的数据
    # 开始保存切分的数据情况
    minvalue,maxvalue=np.min(np.min(gf2data)),np.max(np.max(gf2data)) # 计算最大值和最小值
    
    tifinfo={'minvalue':minvalue,'maxvalue':maxvalue} # 保留数据集的信息
    # 记录每个波段的线性拉伸值
    minvalues=np.array([
        np.percentile(gf2data[:,:,0],2),
        np.percentile(gf2data[:,:,1],2),
        np.percentile(gf2data[:,:,2],2),
        np.percentile(gf2data[:,:,3],2)
    ])
    maxvalues=np.array([
        np.percentile(gf2data[:,:,0],98),
        np.percentile(gf2data[:,:,1],98),
        np.percentile(gf2data[:,:,2],98),
        np.percentile(gf2data[:,:,3],98),
    ])
    tifinfo["band_scalar_minvalue"]=minvalues.tolist()
    tifinfo['band_scale_maxvalue']=maxvalues.tolist()
    saveSpliteResult(traindataset,E512,minvalues,maxvalues) # 保存数据
    tifinfo['minband']=np.min(np.min(gf2data,axis=0),axis=0)[:].tolist() # 最小值
    tifinfo['maxband']=np.max(np.max(gf2data,axis=0),axis=0)[:].tolist() # 最大值
    tifinfo['mean']=np.mean(np.mean(gf2data,axis=0),axis=0)[:].tolist() # 平均值
    tifinfo['std']=np.std(np.mean(gf2data,axis=0),axis=0)[:].tolist() # 标准差
    tifinfo['train']=[]
    tifinfo['test']=[]
    tifinfo['trainnum']=0
    tifinfo['testnum']=0
    tifinfo['NoMaskNum']=0
    for t in E512:
        t['data']=[]
        t['mask']=[]
        if t['sig']==1:
            tifinfo['train'].append(t)
            tifinfo['trainnum']=tifinfo['trainnum']+1
        elif t['sig']==2:
            tifinfo['test'].append(t)
            tifinfo['testnum']=tifinfo['testnum']+1
        else:
            tifinfo['NoMaskNum']=tifinfo['NoMaskNum']+1
    
    tifinfo["exampleNum"]=tifinfo['trainnum']+tifinfo['testnum']+tifinfo['NoMaskNum']
    # 获取比例权重值
    mask_sum=np.sum(mask)/np.sum(mask>-1) # 类别为1 的值
    tifinfo["weight"]=[mask_sum,1-mask_sum]
    # 统计各个波段缩放到255,对应的平均值与标准差
    RGBdata=(gf2data-minvalues)/(maxvalues-minvalues) # 
    RGBdata=RGBdata*255
    RGBdata=np.clip(RGBdata,0,255)
    RGBdata=RGBdata.astype(np.uint8)
    tifinfo['RGBmean']=np.mean(np.mean(RGBdata,axis=0),axis=0)[:].tolist() 
    tifinfo['RGBstd']=np.std(np.mean(RGBdata,axis=0),axis=0)[:].tolist()
    if not os.path.exists(traindataset):
        os.makedirs(traindataset)
    with open(os.path.join(traindataset,"tifinfo.json"),'w',encoding='utf-8') as fp:
        fp.write(json.dumps(tifinfo))
    print('---over-----')

    # 生成代码调试文件，大约有16组对应的训练值，分别对应的 RGB234_test ,ALLBands_test 大小都为256 
    E256_rootdir='/media/gis/databackup/ayc/modellist/dataset/nanchang/E256'
    RGB234_test_path=os.path.join(E256_rootdir,'RGB234_test')
    ALLBands_test_path=os.path.join(E256_rootdir,'ALLBands_test')

    copyDatasetForTest(os.path.join(E256_rootdir,'RGB234'),RGB234_test_path,k_num=16)
    copyDatasetForTest(os.path.join(E256_rootdir,'ALLBands'),ALLBands_test_path,k_num=16)
    print('------copy over--------')