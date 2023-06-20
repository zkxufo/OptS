

export root=/home/l44ye/DATASETS/
export model=mobilenet_v2
echo ${model}
export sens_dir=./SenMap/
GPU_ID=2
qf=98
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                --batchsize 1 \
                                --device "cuda" --root ${root} \
                                --SenMap_dir ${sens_dir} \
                                --OptS_enable True \
                                --Qmax_Y 100 --Qmax_C 100 \
                                --QF_Y ${qf} --QF_C ${qf} \
                                --resize_compress True \


                                