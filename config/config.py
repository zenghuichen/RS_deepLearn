'''
记录训练中的使用的参数
'''
fcn_model_config={
    'datasetName':"Frocodedtest",
    "modelName":"fcn",
    'n_class':2,
    'pre_model_path':None,# 预加载模型
    "learn_rate":0.00001,
    'E512Step':30,
    "maxepoch":200,
    'optimizer':"SGD",
    "scheduler":"ReduceLROnPlateau",
    "lossfunction":{"name":"focalloss","gamma":0, "weight":[0.8641010346111517, 0.13589896538884833], 'size_average':True},#{"name":"cross_entropy"},#{
    "batch_size":8,
    "num_work":8,
    "writerpath":"/home/gis/gisdata/databackup/ayc/modellist/result/writerlog",# writerpath地址路径
    "logpath":"/home/gis/gisdata/databackup/ayc/modellist/result/log",
    'checkpoint':"/home/gis/gisdata/databackup/ayc/modellist/result/checkpoint"
}
