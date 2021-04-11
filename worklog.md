## 版本更新
2021.4.4 00:15
图像增强文件 augmentations.py中，增加了one-hot方法，用于在损失函数中进行计算（主要是Focal Loss,Dice Loss 中）
2021.4.4 08:45
经过检查发现，发现model中fcn模型有问题，因此更换模型来源，将特征提取模型更改为resNet50，为了保证通道数的适配，在原有的resnet的输出上，添加了conv1x1，用于调整通道数，保证模型的适配。
2021.4.4 09:56
通过调整RestNet，fcn 模型已经完成了适配工作。考虑损失函数设计的问题
2021.4.4 16:48
损失函数Focal Loss 代码块
```
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
        inputs=F.softmax(inputs) # 这里使用的softmax 作为类别预测概率，而不是使用sigmoid
        loss_tatal=0
      
        for i in range(gt.shape[1]):
            loss_tatal=loss_tatal-self.weight[i]*torch.pow(1-inputs[:,i,:,:],self.gamma)*torch.log10(inputs[:,i,:,:])*gt[:,i,:,:]    # Focal Loss 累加方法 --直接构建公式
        loss_value=torch.sum(loss_tatal)/1000
        
        if self.reduction=='none':
            return torch.sum(loss_value)
        return loss_value 
```

损失函数验证，模型能够收敛，现在开始尝试调试，训练中的其他的功能：
+ summary 日志记录功能，记录loss，accurary，imageinfo 记录输出
+ log 记录模块记录当前日志输出 
2021.4.5 06:13
fcn模型代码已经完成，从此时开始训练模型
+ RGB123
+ RGB124
+ RGB134
+ RGB234 

2021.4.5 06:16
训练网络模型fcn,具体参数如下:

```
fcn_model_config={
    'datasetName':"RGB123",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':50,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"cross_entropy","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":16,
    "num_work":8,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/result/checkpoint"
}
```
2021.4.5 12:36
调整了最佳模型的保存逻辑，不在256训练阶段保存与评价最优模型。并自此时起，对fcn分别使用RGB123,RGB124,RGB134,RGB234四个数据集，按照下列配置训练模型。
```
fcn_model_config_RGB123={
    'datasetName':"RGB123",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':50,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":16,
    "num_work":8,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/checkpoint"
}

fcn_model_config_RGB124={
    'datasetName':"RGB124",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':50,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":16,
    "num_work":8,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/checkpoint"
}

fcn_model_config_RGB134={
    'datasetName':"RGB134",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':50,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":16,
    "num_work":8,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/checkpoint"
}

fcn_model_config_RGB234={
    'datasetName':"RGB234",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':50,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":16,
    "num_work":8,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/RGB123_fcn_result/checkpoint"
}
```
2021 4.11 10:30 重新调整学习率的变化情况，并且重新训练
2021 4.11 11:06 移除了unet与segnet训练的单独文件，并设置了当loss为nan时，中断当前的出现问题的训练模型与数据集，并继续下一个训练任务。
