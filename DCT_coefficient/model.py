from torchvision import models
from torchvision.models import MobileNet_V2_Weights, AlexNet_Weights
def get_model(Model):

    if Model == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.DEFAULT
        pretrained_model = models.mobilenet_v2(weights=weights).eval()

    elif Model == 'Alexnet':
        weights=AlexNet_Weights.DEFAULT
        pretrained_model = models.alexnet(weights=weights).eval()

    return pretrained_model

