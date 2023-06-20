# Resnet18 Squeezenet Shufflenetv2 Regnet Mnasnet mobilenet_v2 Alexnet
# Regnet800mf Regnet2gf Regnet6gf Regnet8gf Regnet16gf Regnet32gf
GPU_ID=3
for model in Regnet6gf
do
     CUDA_VISIBLE_DEVICES=${GPU_ID} bash ./setting_1.sh ${model}
done
