'''
定义指数集的训练设置
'''
import os 

fcn_model_config_VI={
    'datasetName':"VI",
    "modelName":"fcn",
    'n_class':2,
    'channels':6,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':10,
    "maxepoch":500,
    'optimizer':"SGD",
    "schedule":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2,'scale':10000},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":8,
    "num_work":8,
    'VI_enable':True,
    "writerpath":"./VI_fcn_result/writerlog",# writerpath地址路径
    "logpath":"./VI_fcn_result/log",
    'checkpoint':"./VI_fcn_result/checkpoint"
}
unet_model_config_VI={
    'datasetName':"VI",
    "modelName":"unet",
    'n_class':2,
    'channels':6,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_unet_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':10,
    "maxepoch":500,
    'optimizer':"SGD",
    "schedule":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2,'scale':10000},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":8,
    "num_work":8,
    'VI_enable':True,
    "writerpath":"./VI_unet_result/writerlog",# writerpath地址路径
    "logpath":"./VI_unet_result/log",
    'checkpoint':"./VI_unet_result/checkpoint"
}
segnet_model_config_VI={
    'datasetName':"VI",
    "modelName":"segnet",
    'n_class':2,
    'channels':6,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':10,
    "maxepoch":500,
    'optimizer':"SGD",
    "schedule":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2,'scale':10000},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":8,
    "num_work":8,
    'VI_enable':True,
    "writerpath":"./VI_segnet_result/writerlog",# writerpath地址路径
    "logpath":"./VI_segnet_result/log",
    'checkpoint':"./VI_segnet_result/checkpoint"
}