'''
这里存储主要的数据集结构的
'''

def getRootdirFromDatasetName(datasetName):
    pathlist={
        'RGB432':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB432",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB432",
                    "mean":[930.6206780785358,928.8085351232722,1712.7762373904623],
                    "std":[723.7178269465284,782.6291027983448,1298.1753111629348]},

        'RGB321':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB321",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB321",
                    "mean":[715.9553240254845,930.6206780785358,928.8085351232722],
                    "std":[568.6019165870006,723.7178269465284,782.6291027983448]},

        'RGB421':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB421",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB421",
                    "mean":[715.9553240254845,930.6206780785358,1712.7762373904623],
                    "std":[568.6019165870006,723.7178269465284,1298.1753111629348]},

        'AllBands':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/AllBands",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/AllBands",
                    "mean":[715.9553240254845,930.6206780785358,928.8085351232722,1712.7762373904623],
                    "std":[568.6019165870006,723.7178269465284,782.6291027983448,1712.7762373904623]},    
    }
    return pathlist[datasetName]
