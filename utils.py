from torchvision import models
def load_model(Model):
    if Model=="Alexnet":
        pretrained_model = models.alexnet(pretrained=True).eval()
    elif Model=="Resnet18":
        pretrained_model = models.resnet18(pretrained=True).eval()
    elif Model=="VGG11":  
        pretrained_model = models.vgg11(pretrained=True).eval()
    elif Model=="Squeezenet":
        pretrained_model = models.squeezenet1_0(pretrained=True).eval()
    return pretrained_model
