'''
记录训练中的使用的参数
'''
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
