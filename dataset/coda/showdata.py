'''
展示数据集的切分情况的
'''
from matplotlib import pyplot as plt
import numpy as np
import os



# 首先展示RGB数据集结构
def showImage(tmask,tseg,RGBdata,sig):
    fig=plt.figure(figsize=(800,800))
    plt.subplot(1,4,1)
    plt.imshow(tmask)
    plt.title("mask {}".format(sig))
    plt.subplot(1,4,2)
    plt.imshow(tseg)
    plt.title("seg {}".format(sig))
    plt.subplot(1,4,3)
    tempdata=(RGBdata[:,:,:3]-np.min(RGBdata[:,:,:3]))/(np.max(RGBdata[:,:,:3])-np.min(RGBdata[:,:,:3]))
    plt.imshow(tempdata)
    plt.title("image {}".format(sig))
    plt.subplot(1,4,4)
    mask_mark=np.ones((RGBdata.shape[0],RGBdata.shape[1],3))
    mask_mark[:,:,0]=tseg[:,:]*255
    mask_mark[:,:,1]=RGBdata[:,:,2]/np.max(RGBdata[:,:,2])
    mask_mark[:,:,1]=RGBdata[:,:,2]/np.max(RGBdata[:,:,1])
    plt.imshow(mask_mark)
    plt.title("seg_image {}".format(sig))
    plt.show()    
    pass



def showsigdataset(data_path,sig):
    # 展示3组数据---
    imglist=os.listdir(os.path.join(data_path,"img"))
    i=0
    for name in imglist:
        img=np.load(os.path.join(data_path,"img",name))
        seg=np.load(os.path.join(data_path,"seg",name))
        label=np.load(os.path.join(data_path,"label",name))
        showImage(label,seg,img,sig)
        i=i+1
        if i > 4:
            return 
    

def ShowList(data_path):
    trian_path=os.path.join(data_path,"train")
    test_path=os.path.join(data_path,'test')
    # 数据集结构
    print("train")
    showsigdataset(trian_path,'train')
    print("test")
    showsigdataset(test_path,'test')
    pass

rootdir="/media/gis/databackup/ayc/modellist/dataset/nanchang"

# 首先处理E512
E512_path=os.path.join(rootdir,'E512')
ALLband_path=os.path.join(E512_path,"ALLBands")
RGB123_path=os.path.join(E512_path,"RGB123")
RGB124_path=os.path.join(E512_path,"RGB124")
RGB134_path=os.path.join(E512_path,"RGB134")
RGB234_path=os.path.join(E512_path,"RGB234")

# 开始分批展示
print("E512")
print("ALLBands")
ShowList(ALLband_path)

print("RGB123")
ShowList(RGB123_path)

print("RGB124")
ShowList(RGB124_path)
print("RGB134")
ShowList(RGB134_path)
print("RGB234")
ShowList(RGB234_path)

# 其次展示E256
E256_path=os.path.join(rootdir,'E256')
ALLband_path=os.path.join(E256_path,"ALLBands")
RGB123_path=os.path.join(E256_path,"RGB123")
RGB124_path=os.path.join(E256_path,"RGB124")
RGB134_path=os.path.join(E256_path,"RGB134")
RGB234_path=os.path.join(E256_path,"RGB234")
# 开始分批展示
print("E256")
print("ALLBands")
ShowList(ALLband_path)

print("RGB123")
ShowList(RGB123_path)

print("RGB124")
ShowList(RGB124_path)
print("RGB134")
ShowList(RGB134_path)
print("RGB234")
ShowList(RGB234_path)