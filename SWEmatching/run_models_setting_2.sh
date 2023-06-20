# Resnet18 Squeezenet Shufflenetv2 Regnet Mnasnet mobilenet_v2 Alexnet

GPU_ID=2

for model in Squeezenet Shufflenetv2 Regnet Mnasnet mobilenet_v2 Alexnet
do
    CUDA_VISIBLE_DEVICES=${GPU_ID}  bash setting_2.sh ${model}
done 