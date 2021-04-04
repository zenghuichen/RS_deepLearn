'''
这里存储主要的数据集结构的
'''

# "RGBmean": [98.42966601004996, 103.47446405573739, 89.24539776311728, 118.10080357709782], 
# "RGBstd": [39.68257615485789, 40.96571881718136, 36.232309241791654, 46.13046923546727]

def getRootdirFromDatasetName(datasetName):
    pathlist={
        'RGB123':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB123",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB123",
                    'cls_num':2,
                    "mean":[98.42966601004996, 103.47446405573739, 89.24539776311728],
                    "std":[39.68257615485789, 40.96571881718136, 36.232309241791654]},

        'RGB124':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB124",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB124",
                    'cls_num':2,
                    "mean":[98.42966601004996, 103.47446405573739, 118.10080357709782],
                    "std":[39.68257615485789, 40.96571881718136, 46.13046923546727]},

        'RGB134':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB134",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB134",
                    'cls_num':2,
                    "mean":[98.42966601004996, 89.24539776311728, 118.10080357709782],
                    "std":[39.68257615485789, 36.232309241791654, 46.13046923546727]},

        'RGB234':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB234",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB234",
                    'cls_num':2,
                    "mean":[103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[40.96571881718136, 36.232309241791654, 46.13046923546727]},

        'AllBands':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/AllBands",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/AllBands",
                    "mean":[98.42966601004996, 103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[39.68257615485789, 40.96571881718136, 36.232309241791654, 46.13046923546727]},    


        # 代码开始测试数据集 ，所以不存在512 的数据集，使用256 大小的数据集 代替之
        'RGB234_test':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB234_test",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB234_test",
                    'cls_num':2,
                    "mean":[103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[40.96571881718136, 36.232309241791654, 46.13046923546727]},

        'ALLBands_test':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/ALLBands_test",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/ALLBands_test",
                    'cls_num':2,
                    "mean":[98.42966601004996, 103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[39.68257615485789, 40.96571881718136, 36.232309241791654, 46.13046923546727]},   
    }
    return pathlist[datasetName]
