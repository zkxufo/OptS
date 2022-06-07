from torchvision import models

def get_model(name):
    if name =='Alexnet':
        pretrained_model = models.alexnet(pretrained=True)
    elif name == 'Resnet18':
        pretrained_model = models.resnet18(pretrained=True)
    elif name == 'Squeezenet':
        pretrained_model = models.squeezenet1_0(pretrained=True)
    elif name == 'VGG11':
        pretrained_model = models.vgg11(pretrained=True)
    return pretrained_model
