export root=/home/l44ye/DATASETS/
export model=$1
export sens_dir=./SenMap/
export colorspace=0

# ----------------------------- OptS -------------------------------------

dy_Start=0.005
dy_Step=0.001
dy_End=0.01

# OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dy[i]} ${dc[i]} 
    python OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 50\
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --OptS_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc}

done

dy_Start=0.011
dy_Step=0.01
dy_End=0.1
# OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dy[i]} ${dc[i]} 
    python OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                        --batchsize 50\
                                        --resize_compress True --colorspace ${colorspace} \
                                        --device "cuda" --root ${root} \
                                        --SenMap_dir ${sens_dir} \
                                        --OptS_enable True \
                                        --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                        --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc}

done

dy_Start=0.2
dy_Step=0.1
dy_End=1
# OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dy[i]} ${dc[i]} 
    python OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 50\
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --OptS_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc}

done



# # ----------------------------- OptD -------------------------------------

dy_Start=0.005
dy_Step=0.001
dy_End=0.01
# Fixing d: OptD Matching OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dydc[i]} 
    python3 OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 128 \
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --OptD_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc} 

done

dy_Start=0.011
dy_Step=0.01
dy_End=0.1
# Fixing d: OptD Matching OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dydc[i]} 
    python3 OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 128 \
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --OptD_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc} 

done

dy_Start=0.2
dy_Step=0.1
dy_End=1

# Fixing d: OptD Matching OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dydc[i]} 
    python3 OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 128 \
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --OptD_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc}

done

# # # ----------------------------- JPEG -------------------------------------

dy_Start=0.005
dy_Step=0.001
dy_End=0.01

# Fixing d: JPEG Matching OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dydc[i]} 
    python3 OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 128 \
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --JPEG_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc}

done

dy_Start=0.011
dy_Step=0.01
dy_End=0.1

# Fixing d: JPEG Matching OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dydc[i]} 
    python3 OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 128 \
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --JPEG_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc} 

done

dy_Start=0.2
dy_Step=0.1
dy_End=1

# Fixing d: JPEG Matching OptS
for dydc in $(seq ${dy_Start} ${dy_Step} ${dy_End}); do  
    echo ${dydc[i]} 
    python3 OptD_matching.py --Model ${model} --J 4 --a 4 --b 4 \
                                                            --batchsize 128 \
                                                            --resize_compress True --colorspace ${colorspace} \
                                                            --device "cuda" --root ${root} \
                                                            --SenMap_dir ${sens_dir} \
                                                            --JPEG_enable True \
                                                            --Qmax_Y 100 --Qmax_C 100 --DT_Y 1 --DT_C 1 \
                                                            --d_waterlevel_Y ${dydc} --d_waterlevel_C ${dydc}

done



