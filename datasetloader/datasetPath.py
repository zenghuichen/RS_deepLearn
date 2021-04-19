'''
这里存储主要的数据集结构的
'''
'''统计RGB'''
# "RGBmean": [98.42966601004996, 103.47446405573739, 89.24539776311728, 118.10080357709782], 
# "RGBstd": [39.68257615485789, 40.96571881718136, 36.232309241791654, 46.13046923546727]
''' 统计遥感影像 '''
# "allbandmean":[715.9553240254821, 930.6206780785278, 928.8085351232803, 1712.776237390446]
# "allbandstd":[292.5454016033627, 372.57264030177237, 381.12971420208686, 673.9235916158207]
import sys
def getRootdirFromDatasetName(datasetName):
    pathlist_win32={
        'RGB123':{'E256':r"E:\zenghui\dataset\nanchang\E256\RGB123",
                    "E512":r"E:\zenghui\dataset\nanchang\E512\RGB123",
                    'cls_num':2,
                    'channels':3,
                    "mean":[98.42966601004996, 103.47446405573739, 89.24539776311728],
                    "std":[39.68257615485789, 40.96571881718136, 36.232309241791654]},

        'RGB124':{'E256':r"E:\zenghui\dataset\nanchang\E256\RGB124",
                    "E512":r"E:\zenghui\dataset\nanchang\E512\RGB124",
                    'cls_num':2,
                    'channels':3,
                    "mean":[98.42966601004996, 103.47446405573739, 118.10080357709782],
                    "std":[39.68257615485789, 40.96571881718136, 46.13046923546727]},

        'RGB134':{'E256':r"E:\zenghui\dataset\nanchang\E256\RGB134",
                    "E512":r"E:\zenghui\dataset\nanchang\E512\RGB134",
                    'cls_num':2,
                    'channels':3,
                    "mean":[98.42966601004996, 89.24539776311728, 118.10080357709782],
                    "std":[39.68257615485789, 36.232309241791654, 46.13046923546727]},

        'RGB234':{'E256':r"E:\zenghui\dataset\nanchang\E256\RGB234",
                    "E512":r"E:\zenghui\dataset\nanchang\E512\RGB234",
                    'cls_num':2,
                    'channels':3,
                    "mean":[103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[40.96571881718136, 36.232309241791654, 46.13046923546727]},

        'AllBands':{'E256':r"E:\zenghui\dataset\nanchang\E256\ALLBands",
                    "E512":r"E:\zenghui\dataset\nanchang\E512\ALLBands",
                    'cls_num':2,
                    'channels':3,
                    'min':-1262.0,
                    'max':9483.0,
                    "mean":[715.9553240254821, 930.6206780785278, 928.8085351232803, 1712.776237390446],
                    "std":[292.5454016033627, 372.57264030177237, 381.12971420208686, 673.9235916158207]},    
        'VI':{'E256':r"E:\zenghui\dataset\nanchang\E256\ALLBands",
                    "E512":r"E:\zenghui\dataset\nanchang\E512\ALLBands",
                    'cls_num':2,
                    'channels':3,
                    'min':-1262.0,
                    'max':9483.0,
                    "mean":[715.9553240254821, 930.6206780785278, 928.8085351232803, 1712.776237390446],
                    "std":[292.5454016033627, 372.57264030177237, 381.12971420208686, 673.9235916158207]},    

        # 代码开始测试数据集 ，所以不存在512 的数据集，使用256 大小的数据集 代替之
        'RGB234_test':{'E256':r"E:\zenghui\dataset\nanchang\E256\RGB234_test",
                    "E512":r"E:\zenghui\dataset\nanchang\E256\RGB234_test",
                    'cls_num':2,
                    'channels':3,
                    "mean":[103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[40.96571881718136, 36.232309241791654, 46.13046923546727]},

        'ALLBands_test':{'E256':r"E:\zenghui\dataset\nanchang\E256\ALLBands_test",
                    "E512":r"E:\zenghui\dataset\nanchang\E256\ALLBands_test",
                    'cls_num':2,
                    'channels':4,
                    "mean":[98.42966601004996, 103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[39.68257615485789, 40.96571881718136, 36.232309241791654, 46.13046923546727]},   
    }

    pathlist_linus= {
        'RGB123':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB123",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB123",
                    'cls_num':2,
                    'channels':3,
                    "mean":[98.42966601004996, 103.47446405573739, 89.24539776311728],
                    "std":[39.68257615485789, 40.96571881718136, 36.232309241791654]},

        'RGB124':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB124",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB124",
                    'cls_num':2,
                    'channels':3,
                    "mean":[98.42966601004996, 103.47446405573739, 118.10080357709782],
                    "std":[39.68257615485789, 40.96571881718136, 46.13046923546727]},

        'RGB134':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB134",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB134",
                    'cls_num':2,
                    'channels':3,
                    "mean":[98.42966601004996, 89.24539776311728, 118.10080357709782],
                    "std":[39.68257615485789, 36.232309241791654, 46.13046923546727]},

        'RGB234':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB234",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB234",
                    'cls_num':2,
                    'channels':3,
                    "mean":[103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[40.96571881718136, 36.232309241791654, 46.13046923546727]},

        'AllBands':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/ALLBands",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/ALLBands",
                    'cls_num':2,
                    'channels':3,
                    'min':-1262.0,
                    'max':9483.0,
                    "mean":[715.9553240254821, 930.6206780785278, 928.8085351232803, 1712.776237390446],
                    "std":[292.5454016033627, 372.57264030177237, 381.12971420208686, 673.9235916158207]},    
        'VI':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/ALLBands",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/ALLBands",
                    'cls_num':2,
                    'channels':3,
                    'min':-1262.0,
                    'max':9483.0,
                    "mean":[715.9553240254821, 930.6206780785278, 928.8085351232803, 1712.776237390446],
                    "std":[292.5454016033627, 372.57264030177237, 381.12971420208686, 673.9235916158207]},    

        # 代码开始测试数据集 ，所以不存在512 的数据集，使用256 大小的数据集 代替之
        'RGB234_test':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB234_test",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB234_test",
                    'cls_num':2,
                    'channels':3,
                    "mean":[103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[40.96571881718136, 36.232309241791654, 46.13046923546727]},

        'ALLBands_test':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/ALLBands_test",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/ALLBands_test",
                    'cls_num':2,
                    'channels':4,
                    "mean":[98.42966601004996, 103.47446405573739, 89.24539776311728, 118.10080357709782],
                    "std":[39.68257615485789, 40.96571881718136, 36.232309241791654, 46.13046923546727]},   
    }
    if sys.platform == "linux":
        return pathlist_linus[datasetName]
    elif sys.platform=="win32":
        return pathlist_win32[datasetName]
