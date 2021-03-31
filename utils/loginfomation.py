import os
import numpy as np
import pandas as pds


class loginfomation(object):
    def __init__(self,model_name,rootdir):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        self.model_name=model_name
        self.csvpath=os.path.join(rootdir,"val_log_{}.csv".format(model_name))
        self.logtrainpath=os.path.join(rootdir,"trian_log_{}.txt".format(model_name))
        self.logvalpath=os.path.join(rootdir,"val_log_{}.txt".format(model_name))
        self.logtestpath=os.path.join(rootdir,"test_log_{}.txt".format(model_name))
        
        self.loginfo={"modelname":[],'epoch':[]}

    def logValInfo(self,epoch,scores_val,class_iou_val):
        self.loginfo["epoch"].append(epoch)
        self.loginfo['modelname'].append(self.model_name)
        for k, v in scores_val.items():
            if k not in self.loginfo:
                self.loginfo[k]=[]
            self.loginfo[k].append(v)
        for k, v in class_iou_val.items():
            if k not in self.loginfo:
                self.loginfo[k]=[]
            self.loginfo[k].append(v) 
        # 输出至普通文件
        csvobj=pds.DataFrame(self.loginfo)
        csvobj.to_csv(self.csvpath,encoding='utf-8')

    def logTrainlog(self,stringline):
        with open(self.logtrainpath,'a',encoding='utf-8') as fp:
            for lines in stringline:
                fp.write("{}\n".format(lines))
    
    def logvallog(self,stringline):
        with open(self.logvalpath,'a',encoding='utf-8') as fp:
            for lines in stringline:
                fp.write("{}\n".format(lines))
    
    def logtestlog(self,stringline):
        with open(self.logtestpath,'a',encoding='utf-8') as fp:
            for lines in stringline:
                fp.write("{}\n".format(lines))    