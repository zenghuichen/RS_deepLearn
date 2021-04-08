from models.fcn import *
from models.segnet import *
from models.unet import *
def get_models(modelname,num_class=2):
    if modelname=="fcn":
        resnet50_model=ResNet50()
        model=FCNs(pretrained_net=resnet50_model, n_class=num_class)
    elif modelname=='segnet':
        model=segnet(n_classes=num_class)
    elif modelname=='unet':
        model=UNet(n_classes=num_class)
    return model