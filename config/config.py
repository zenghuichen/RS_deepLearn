'''
记录训练中的使用的参数
'''
fcn_model_config={
    'datasetName':"RGB234",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,# 预加载模型
    "learn_rate":0.1,
    'E512Step':30,
    "maxepoch":200,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"cross_entropy","weight":None,'reduction':'mean','gamma':2},#{"name":"focalloss","gamma":0, "weight":[1,10], 'size_average':True},
    "batch_size":8,
    "num_work":8,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/result/checkpoint"
}
