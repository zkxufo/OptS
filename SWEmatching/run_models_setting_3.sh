# mobilenet_v2 Regnet400mf Resnet18 Shufflenetv2 Resnet18 Squeezenet  Mnasnet Alexnet
GPU_ID=2
for model in  mobilenet_v2 Regnet400mf
do
    CUDA_VISIBLE_DEVICES=${GPU_ID}  bash setting_3.sh ${model}
done 