# Resnet18 Squeezenet Shufflenetv2 Regnet Mnasnet mobilenet_v2 Alexnet
# Regnet800mf Regnet2gf Regnet6gf Regnet8gf Regnet16gf Regnet32gf
GPU_ID=2
# mobilenet_v2 Regnet400mf Resnet18 Squeezenet Shufflenetv2 Mnasnet Alexnet
for model in mobilenet_v2 Regnet400mf Resnet18 Squeezenet Shufflenetv2 Mnasnet Alexnet
do
     CUDA_VISIBLE_DEVICES=${GPU_ID} bash ./setting_1.sh ${model}
done
