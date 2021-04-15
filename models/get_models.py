from models.fcn import *
from models.segnet import *
from models.unet import *
def get_models(modelname,num_class=2,channels_num=3):
    if modelname=="fcn":
        resnet50_model=ResNet50(input_channels=channels_num)
        model=FCNs(pretrained_net=resnet50_model, n_class=num_class)
    elif modelname=='segnet':
        model=segnet(n_classes=num_class,in_channels=channels_num)
    elif modelname=='unet':
        model=UNet(n_classes=num_class,n_channels=channels_num)
    return model