from models.fcn import *
from models.segnet import *
from models.unet import *

def get_models(modelname,num_class=2):
    if modelname=="fcn":
        model=fcn32s(n_classes=num_class)
    elif modelname=='segnet':
        model=segnet(n_classes=num_class)
    elif modelname=='unet':
        model=unet(n_classes=num_class)
    
    return model