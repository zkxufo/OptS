export root=/home/l44ye/DATASETS/
export model=$1
export sens_dir=./SenMap/
export colorspace=0
GPU_ID=2

# ----------------------------- OptS -------------------------------------

dy_Start=0.005
dy_Step=0.001
dy_End=0.01

dc_Start=0.011
dc_Step=0.01
dc_End=0.1

# for testing
# dy_Start=0.5
# dy_Step=0.001
# dy_End=0.5

# dc_Start=0.5
# dc_Step=0.01
# dc_End=0.5

# OptS
for dy in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    for dc in $(seq ${dc_Start} ${dc_Step} ${dc_End}); do  
            echo ${dy[i]} ${dc[i]} 
            CUDA_VISIBLE_DEVICES=${GPU_ID} python OptS.py --Model ${model} --J 4 --a 4 --b 4 \
                                                                    --batchsize 50\
                                                                    --resize_compress True --colorspace ${colorspace} \
                                                                    --device "cuda" --root ${root} \
                                                                    --SenMap_dir ${sens_dir} \
                                                                    --OptS_enable True \
                                                                    --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                                    --d_waterlevel_Y ${dy} --d_waterlevel_C ${dc}
    done
done

