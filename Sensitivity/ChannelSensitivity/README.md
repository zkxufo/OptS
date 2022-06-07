`# Run optimal color space conversion
### Alexnet
python3 ChannelSen.py -model Alexnet -Batch_size 100 -Nexample 10000
### VGG11
python3 ChannelSen.py -model VGG11 -Batch_size 100 -Nexample 10000
### Squeezenet
python3 ChannelSen.py -model Squeezenet -Batch_size 100 -Nexample 10000
### Resnet18
python3 ChannelSen.py -model Resnet18 -Batch_size 100 -Nexample 10000

# Run Grace
### Alexnet
python3 ChannelSenSTD.py -model Alexnet -Batch_size 100 -Nexample 10000 -grace
### VGG11
python3 ChannelSenSTD.py -model VGG11 -Batch_size 100 -Nexample 10000 -grace
### Squeezenet
python3 ChannelSenSTD.py -model Squeezenet -Batch_size 100 -Nexample 10000 -grace
### Resnet18
python3 ChannelSenSTD.py -model Resnet18 -Batch_size 100 -Nexample 10000 -grace

# Run standard color space conversion
### Alexnet
python3 ChannelSenSTD.py -model Alexnet -Batch_size 100 -Nexample 10000
### VGG11
python3 ChannelSenSTD.py -model VGG11 -Batch_size 100 -Nexample 10000
### Squeezenet
python3 ChannelSenSTD.py -model Squeezenet -Batch_size 100 -Nexample 10000
### Resnet18
python3 ChannelSenSTD.py -model Resnet18 -Batch_size 100 -Nexample 10000

# Without color space conversion
### Alexnet
python3 ChannelSenRGB.py -model Alexnet -Batch_size 100 -Nexample 10000
### VGG11
python3 ChannelSenRGB.py -model VGG11 -Batch_size 100 -Nexample 10000
### Squeezenet
python3 ChannelSenRGB.py -model Squeezenet -Batch_size 100 -Nexample 10000
### Resnet18
python3 ChannelSenRGB.py -model Resnet18 -Batch_size 100 -Nexample 10000
