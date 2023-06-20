from torchvision import models
from torchvision.models import MobileNet_V2_Weights, AlexNet_Weights
def get_model(Model):
    if Model=="Resnet18":
        pretrained_model = models.resnet18(pretrained=True).eval()
    if Model=="Resnet34":
        pretrained_model = models.resnet34(pretrained=True).eval()
    if Model=="Resnet50":
        pretrained_model = models.resnet50(pretrained=True).eval()
    if Model=="Resnet101":
        pretrained_model = models.resnet101(pretrained=True).eval()
    if Model=="Resnet152":
        pretrained_model = models.resnet152(pretrained=True).eval()
    elif Model=="Squeezenet":
        pretrained_model = models.squeezenet1_0(pretrained=True).eval()

    elif Model == 'Shufflenetv2_05':
        pretrained_model = models.shufflenet_v2_x0_5(pretrained=True).eval()
    elif Model == 'Shufflenetv2_10':
        pretrained_model = models.shufflenet_v2_x1_0(pretrained=True).eval()
    elif Model == 'Shufflenetv2_15':
        pretrained_model = models.shufflenet_v2_x1_5(pretrained=True).eval()
    elif Model == 'Shufflenetv2_20':
        pretrained_model = models.shufflenet_v2_x2_0(pretrained=True).eval()
    
    elif Model == 'Mnasnet':
        pretrained_model = models.mnasnet1_0(pretrained=True).eval()
    elif Model == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.DEFAULT
        pretrained_model = models.mobilenet_v2(weights=weights).eval()
    elif Model == 'efficientnet_b1':
        pretrained_model = models.efficientnet_b1(pretrained=True).eval()
    elif Model == 'Alexnet':
        weights=AlexNet_Weights.DEFAULT
        pretrained_model = models.alexnet(weights=weights).eval()
    

    elif Model == 'convnext_base':
        weights=models.ConvNeXt_Base_Weights.DEFAULT
        pretrained_model = models.convnext_base(weights=weights).eval()
    elif Model == 'convnext_tiny':
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        pretrained_model = models.convnext_tiny(weights=weights).eval()
    elif Model == 'convnext_large':
        weights=models.ConvNeXt_Large_Weights.DEFAULT
        pretrained_model = models.convnext_large(weights=weights).eval()
    elif  Model == 'convnext_small':
        weights=models.ConvNeXt_Small_Weights.DEFAULT
        pretrained_model = models.convnext_small(weights=weights).eval()
    
    elif Model == 'Regnet400mf':
        pretrained_model = models.regnet_y_400mf(pretrained=True).eval()
    elif Model == 'Regnet800mf':
        pretrained_model = models.regnet_y_800mf(pretrained=True).eval()
    elif Model == 'Regnet6gf':
        pretrained_model = models.regnet_y_1_6gf(pretrained=True).eval()
    elif Model == 'Regnet2gf':
        pretrained_model = models.regnet_y_3_2gf(pretrained=True).eval()
    elif Model == 'Regnet8gf':
        pretrained_model = models.regnet_y_8gf(pretrained=True).eval()
    elif Model == 'Regnet16gf':
        pretrained_model = models.regnet_y_16gf(pretrained=True).eval()
    elif Model == 'Regnet32gf':
        pretrained_model = models.regnet_y_32gf(pretrained=True).eval()


    return pretrained_model

