export root=/home/l44ye/DATASETS/
# Resnet18 Squeezenet Shufflenetv2 Regnet Mnasnet mobilenet_v2 Alexnet
export sens_dir=./SenMap/
GPU_ID=0

# # Fixing QF: OptS Matching JPEG 
# Regnet400mf Regnet800mf Regnet2gf Regnet6gf Regnet8gf Regnet16gf Regnet32gf
# for model in Regnet400mf Regnet800mf Regnet2gf Regnet6gf
for model in Regnet6gf Regnet2gf   
do
    for qf in `seq 98 -1 70`
    do
        echo Cross Model with a base Senstivity : ${model}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 HDQ_matching_crossModel.py --Model ${model} --J 4 --a 4 --b 4 \
                                        --batchsize 10 \
                                        --device "cuda" --root ${root} \
                                        --SenMap_dir ${sens_dir} \
                                        --OptS_enable True \
                                        --Qmax_Y 100 --Qmax_C 100 \
                                        --QF_Y ${qf} --QF_C ${qf} \
                                        --resize_compress True \
                                        
    done
done


for model in Regnet800mf   
do
    for qf in `seq 86 -1 70`
    do
        echo Cross Model with a base Senstivity : ${model}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 HDQ_matching_crossModel.py --Model ${model} --J 4 --a 4 --b 4 \
                                        --batchsize 10 \
                                        --device "cuda" --root ${root} \
                                        --SenMap_dir ${sens_dir} \
                                        --OptS_enable True \
                                        --Qmax_Y 100 --Qmax_C 100 \
                                        --QF_Y ${qf} --QF_C ${qf} \
                                        --resize_compress True \
                                        
    done
done