'''
这里存储主要的数据集结构的
'''

def getRootdirFromDatasetName(datasetName):
    pathlist={
        'RGB432':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB432",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB432"},

        'RGB321':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB321",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB321"},

        'RGB421':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/RGB421",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/RGB421"},

        'AllBands':{'E256':"/media/gis/databackup/ayc/modellist/dataset/nanchang/E256/AllBands",
                    "E512":"/media/gis/databackup/ayc/modellist/dataset/nanchang/E512/AllBands"},    
    }
    return pathlist[datasetName]
