export root=/home/l44ye/DATASETS/
# Resnet18 Squeezenet Shufflenetv2 Regnet Mnasnet mobilenet_v2 Alexnet
export sens_dir=./SenMap/
GPU_ID=2

export model=mobilenet_v2
echo ${model}

# Remove it for normal inference --> compress_resize
# CUDA_VISIBLE_DEVICES=${GPU_ID}  python HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
# 								--batchsize 10 \
# 								--device "cuda" --root ${root} \
# 								--SenMap_dir ${sens_dir} \
# 								--OptS_enable True \
# 								--Qmax_Y 100 --Qmax_C 100 


# Fixing QF: OptS Matching JPEG 
dy_Start=0.5
dy_Step=0.001
dy_End=0.5
colorspace=0
# Fixing d: JPEG Matching OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dydc[i]} 
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 50 \
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --OptD_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc} 

done