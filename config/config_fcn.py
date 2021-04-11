'''
记录训练中的使用的参数
'''

'''fcn'''
fcn_model_config_RGB123={
    'datasetName':"RGB123",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':20,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2,'scale':10000},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":32,
    "num_work":32,
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
    'E512Step':20,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2,'scale':10000},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":32,
    "num_work":32,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB124_fcn_result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB124_fcn_result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/RGB124_fcn_result/checkpoint"
}

fcn_model_config_RGB134={
    'datasetName':"RGB134",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':20,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2,'scale':10000},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":32,
    "num_work":32,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB134_fcn_result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB134_fcn_result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/RGB134_fcn_result/checkpoint"
}

fcn_model_config_RGB234={
    'datasetName':"RGB234",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,#'/media/gis/databackup/ayc/modellist/result/checkpoint/ckpt_fcn_epoch_860.pth',# 预加载模型
    "learn_rate":0.01,
    'E512Step':20,
    "maxepoch":500,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"FocalLoss","weight":[0.023296172671643,0.976703827328357],'reduction':'sum','gamma':2,'scale':10000},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":32,
    "num_work":32,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB234_fcn_result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/RGB234_fcn_result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/RGB234_fcn_result/checkpoint"
}