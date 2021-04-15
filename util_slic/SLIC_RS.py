'''
使用 SLIC 完成图像的 超像素分割
读取遥感图像
参照：https://www.cnblogs.com/ninicwang/p/11533066.html

# 注意处理过的影像数据
1 ： mask
2-5 : 数据波段
'''

import os
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt
import math
from collections import Counter
from matplotlib import cm


def ReadRSwithMask(gfpath,maskidx):
    '''数据文件地址'''
    # 读取tiff文件
    dataset=gdal.Open(gfpath)
    im_width=dataset.RasterXSize # 列数
    im_height=dataset.RasterYSize # 行数
    im_bands=dataset.RasterCount # 波段数
    im_geotrans=dataset.GetGeoTransform() # 仿射矩阵
    im_proj=dataset.GetProjection() # 地图投影信息

    maskRS=dataset.GetRasterBand(maskidx) # 
    mask=maskRS.ReadAsArray(0,0,im_width,im_height)

    RSimg=np.ones((mask.shape[0],mask.shape[1],im_bands))
    for i in range(0,im_bands):
        band1=dataset.GetRasterBand(i+1)
        band1_arr=band1.ReadAsArray(0,0,im_width,im_height)
        minb1=np.min(band1_arr)
        maxb1=np.max(band1_arr)
        # 归一化
        band1arr=(band1_arr-minb1)/(maxb1-minb1)

        RSimg[:,:,i]=band1arr
    
    # 清理mask 部分的代码
    RSimg=np.delete(RSimg,maskidx-1,axis=2)
    return RSimg,mask,im_bands

def NearSeed(gradient,i,j):
    h,w=gradient.shape
    i_start=int(i-1) if i-1>=0 else 0
    i_end=int(i+1) if i+1<h else int(h-1)
    j_start=int(j-1) if j-1>=0 else 0
    j_end=int(j+1) if j+1<w else int(w-1)

    dg=math.inf
    ij=[]
    for i in range(i_start,i_end+1):
        for j  in range(j_start,j_end+1):
            if dg>gradient[i,j]:
                ij=[i,j]
    return i,j

def InitSeed(RSimg,S,bandnum=4):
    # 返回 种子seeds的值
    h=RSimg.shape[0]
    w=RSimg.shape[1]
    ilist=np.array(list(range(0,h,int(S))))
    jlist=np.array(list(range(0,w,int(S))))
    xyidx=Cartesianproduct(ilist,jlist) # [i,j]
    # 获取 seed 的 参数值
    # seedsband=np.zeros((xyidx.shape[0],bandnum))
    seedsband=RSimg[xyidx[:,0],xyidx[:,1],:]
    seedsxy=xyidx
    seeds=np.concatenate((seedsband,seedsxy),axis=1)
    # 调整种子到周围3x3 领域内 梯度最小的区域
    gradient=np.gradient(RSimg)
    gradientx=gradient[0] # x
    gradienty=gradient[1] # y 
    gradientxy=np.sum(np.sqrt(np.power(gradientx,2)+np.power(gradienty,2)),axis=2)
    # 进行局部调整 梯度最小的值
    for i in range(seeds.shape[0]):
        seed=seeds[i,:]
        # 选择周围像素最小
        ii,jj=NearSeed(gradientxy,int(seed[-2]),int(seed[-1]))
        RSband=RSimg[ii,jj,:].reshape(1,-1)
        iijj=np.array([ii,jj]).reshape(1,-1)
        RS=np.concatenate((RSband,iijj),axis=1)
        seeds[i,:]=RS
    return seeds 

def Cartesianproduct(x,y):
    # 笛卡尔乘积
    # x 宽
    # y 高
    x=x.reshape((x.shape[0],1))
    y=y.reshape((1,y.shape[0]))
    xy=np.ones((x.shape[0],y.shape[1],2))
    xy[:,:,0]=xy[:,:,0]*x
    xy[:,:,1]=xy[:,:,1]*y
    xyidx=xy.reshape(xy.shape[0]*xy.shape[1],2)
    xyidx=xyidx.astype(np.int)
    return xyidx

def Distance(pixels,seed,m,s,bandnum=4):
    # pixel 数据格式 [b0,b1,b2,b3,x,y]
    # seed [b0,b1,b2,b3,x,y]
    # m 超参数
    # s=sqrt(N/K) N 像素数量 ，K seed 数量
    d=pixels-seed
    dis=np.power(d,2)
    disband=dis[:,list(range(bandnum))]
    disband=np.sqrt(np.sum(disband,axis=1))
    disxy=dis[:,list(range(bandnum,bandnum+2))]
    disxy=np.sqrt(np.sum(disxy,axis=1))
    Ds=disband+(m/s)*disxy
    return Ds

def ComputeNearSeed(RSimg,seed,labelarr,disarr,S,m,seedlabel,bandnum=4):
    # 计算周围临近的像素
    h,w,bs=RSimg.shape
    i_start=int(seed[0,bandnum]-S) if seed[0,bandnum]-S >=0 else 0
    j_start=int(seed[0,bandnum+1]-S) if seed[0,bandnum+1]-S >=0 else 0
    i_end=int(seed[0,bandnum]+S) if seed[0,bandnum]+S <h else int(h-1)
    j_end=int(seed[0,bandnum+1]+S) if seed[0,bandnum+1]+S <w else int(w-1)
    # 裁剪范围阵进行计算
    ilist=np.array(list(range(i_start,i_end+1,1))).astype(np.int)
    jlist=np.array(list(range(j_start,j_end+1,1))).astype(np.int)
    ijls=Cartesianproduct(ilist,jlist) # [i,j]
    RSNbs=RSimg[ijls[:,0],ijls[:,1],:] # 选中波段
    # 计算distance
    RSN=np.concatenate((RSNbs,ijls),axis=1)
    Ds=Distance(RSN,seed,m,S,bandnum=bandnum)
    # 选中 distance ， label
    disNarr=disarr[ijls[:,0],ijls[:,1]] # 选中距离
    labelNarr=labelarr[ijls[:,0],ijls[:,1]] # 计算标签
    disN=disNarr-Ds
    idx=np.where(disN>0)
    labelNarr[idx]=seedlabel # 标记类别
    disNarr[idx]=Ds[idx]
    # 将数据还原回去
    disarr[ijls[:,0],ijls[:,1]]=disNarr
    labelarr[ijls[:,0],ijls[:,1]]=labelNarr
    return disarr,labelarr


def ComputerSeed(RSimg,seeds,labelarr,disarr,S,m,bandnum=4):
    for i in range(seeds.shape[0]):
        seed=seeds[i,:] # 获取种子
        seed=seed.reshape(1,-1)
        disarr,labelarr=ComputeNearSeed(RSimg,seed,labelarr,disarr,S,m,i,bandnum=bandnum)
    return disarr,labelarr


def UpdataSeed(RSimg,labelarr):
    lbs=np.unique(labelarr)
    xy=np.ones((RSimg.shape[0],RSimg.shape[1],2))
    ils=np.array(list(range(xy.shape[0]))).reshape(-1,1)
    jls=np.array(list(range(xy.shape[1]))).reshape(1,-1)
    xy[:,:,0]=xy[:,:,0]*ils
    xy[:,:,1]=xy[:,:,1]*jls
    RS=np.concatenate((RSimg,xy),axis=2)
    labels=np.unique(labelarr).tolist()
    seeds=np.zeros((len(labels),RS.shape[2]))
    for label in labels:
        idx=np.where(labelarr==label)
        RSlabel=RS[idx[0],idx[1],:]
        # 计算中
        RScenter=np.mean(RSlabel,axis=0).reshape(1,-1)
        seeds[label,:]=RScenter  
    return seeds


def filterNear(labelarr,i,j,size=3):
    h,w=labelarr.shape
    i_start=int(i-1) if i-1>=0 else 0
    i_end=int(i+1) if i+1<h else h-1
    j_start=int(j-1) if j-1>=0 else 0
    j_end=int(j+1) if j+1<w else w-1
    lb=labelarr[i,j]
    lbs=[] 
    for ii in range(i_start,i_end+1):
        for jj in range(j_start,j_end+1):
            lbs.append(labelarr[ii,jj])
    lbcount=dict(Counter(lbs))
    if lbcount[lb]==1 and len(list(lbcount.keys()))==2:
        lbt=list(lbcount.keys())
        for l in lbt:
            if l!=lb:
                labelarr[i,j]=l
    return labelarr


def findEdge(labelarr,i,j):
    h,w=labelarr.shape
    i_start=int(i-1) if i-1>=0 else 0
    i_end=int(i+1) if i+1<h else h-1
    j_start=int(j-1) if j-1>=0 else 0
    j_end=int(j+1) if j+1<w else w-1
    lb=labelarr[i,j]
    lbs=[] 
    for ii in range(i_start,i_end+1):
        for jj in range(j_start,j_end+1):
            lbs.append(labelarr[ii,jj])
    lbcount=dict(Counter(lbs))
    for i in lbcount:
        if i==lb and lbcount[i]==1:
            return 0
    if len(lbcount.keys())>1:
        return 1



def markEdge(labelarr):
    # 平滑滤波
    lbs=labelarr.reshape(-1).tolist()
    lbc=dict(Counter(lbs)) # 统计细碎图像
    lbcarr=np.zeros((len(lbc.values()),2))
    for i in lbc:
        lbcarr[i,0]=i
        lbcarr[i,1]=lbc[i]
    minv=np.min(lbcarr[:,1])
    # 

    # 声明边界板
    mark=np.zeros((labelarr.shape[0],labelarr.shape[1]))
    for i in range(labelarr.shape[0]):
        for j in range(labelarr.shape[1]):
            mark[i,j]=findEdge(labelarr,i,j)
    return mark


def addMaskAndMark(maskline,markline):
    maskline=maskline*255
    markline=markline*255
    mark=np.zeros((markline.shape[0],markline.shape[1],3))
    mark[:,:,0]=maskline
    mark[:,:,1]=markline
    mark=mark.astype(np.uint8)
    return mark


def RS_SLIC(RSimg,k,m,iter,MErr,bandnum=4,isdrawing=False,mask=None,gfdir=None):
    N=RSimg.shape[0]*RSimg.shape[1] # 像素数量
    S=math.sqrt(N/k)
    seeds=InitSeed(RSimg,S,bandnum=bandnum) # 返回种子 seeds 
    print("s:{},m:{}".format(S,m))
    for i in range(iter):
        labelarr=-1*np.ones((RSimg.shape[0],RSimg.shape[1]),dtype=np.int)
        disarr=np.ones((RSimg.shape[0],RSimg.shape[1]))*math.inf
        disarr,labelarr=ComputerSeed(RSimg,seeds,labelarr,disarr,S,m)
        seeds=UpdataSeed(RSimg,labelarr) # 更新种子中心
        if isdrawing:
            print("绘制图像 {} ".format(i))
            mark=markEdge(labelarr)
            mark=addMaskAndMark(mask,mark)
            fig=plt.figure(figsize=(13,13))
            plt.cla()
            plt.matshow(labelarr,cmap =cm.prism)
            plt.show()
            fig.savefig(os.path.join(gfdir,'label{}.jpg'.format(i)),dpi=300)
            fig=plt.figure(figsize=(13,13))
            plt.cla()
            plt.imshow(mark)
            plt.show()
            fig.savefig(os.path.join(gfdir,'mark_{}.png'.format(i)), dpi=300)

    labelarr=filter(labelarr)
    # 计算标签的边界
    mark=markEdge(labelarr)
    return labelarr,mark

if __name__ == "__main__":
    gfdir='/home/gis/gisdata/data/ayc/superpixel_fcn/RSImage/megerMASK'
    gfpath=os.path.join(gfdir,'andbandsmall.tif')
    # gfdir='/home/gis/gisdata/data/ayc/superpixel_fcn/RSImage/jietu'
    # gfpath=os.path.join(gfdir,'321.png')
    rsimg,mask,im_bands=ReadRSwithMask(gfpath,1)
    maskline=markEdge(mask)
    m=40
    labelarr,markline=RS_SLIC(rsimg*255,49,m,10,0.5,bandnum=im_bands,isdrawing=True,mask=maskline,gfdir=gfdir)
    maskline=maskline*255
    markline=markline*255
    mark=np.zeros((markline.shape[0],markline.shape[1],3))
    mark[:,:,0]=maskline
    mark[:,:,1]=markline
    mark=mark.astype(np.uint8)


    markline=markline==0
    markline=markline*1
    markline=markline.astype(np.int)
    trs=rsimg.copy()
    for i in range(trs.shape[2]):
        trs[:,:,i]=trs[:,:,i]*markline
    trs=trs*255
    trs=trs.astype(np.uint8)
    fig=plt.figure(figsize=(13,13))
    plt.cla()
    plt.matshow(labelarr,cmap = plt.cm.prism)
    plt.show()
    fig.savefig(os.path.join(gfdir,'label.jpg'))
    fig=plt.figure(figsize=(13,13))
    plt.cla()
    plt.imshow(mark)
    plt.show()
    fig.savefig(os.path.join(gfdir,'mark.jpg'))
    fig=plt.figure(figsize=(20,20))
    plt.cla()
    plt.imshow(trs[:,:,[1,2,3]])
    plt.show()
    fig.savefig(os.path.join(gfdir,'trs.jpg'))
    fig=plt.figure(figsize=(16,16))
    plt.cla()
    plt.imshow(rsimg[:,:,[2,3,0]])
    plt.show()
    fig.savefig(os.path.join(gfdir,'rsimg3.jpg'))