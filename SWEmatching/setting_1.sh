export root=~/Imagenet/
export model=mobilenet_v2
export sens_dir=./SenMap/


# Fixing QF: OptS Matching JPEG 
for qf in `seq 98 -1 70`
do
	python3.10 HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
									--batchsize 50 \
									--device "cuda" --root ${root} \
									--output_txt ${file} \
									--SenMap_dir ${sens_dir} \
									--resize_compress True \
									--OptS_enable True \
									--Qmax_Y 100 --Qmax_C 100 \
									--QF_Y ${qf} --QF_C ${qf} 
									
done


# Fixing QF: OptD Matching JPEG 
for qf in `seq 98 -1 70`
do
	python3.10 HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
									--batchsize 50 \
									--device "cuda" --root ${root} \
									--output_txt ${file} \
									--SenMap_dir ${sens_dir} \
									--resize_compress True \
									--OptD_enable True \
									--Qmax_Y 100 --Qmax_C 100 \
									--QF_Y ${qf} --QF_C ${qf} 
done

# Normal JEPG
for qf in `seq 98 -1 70`
do
	python3.10 HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
									--batchsize 128 \
									--device "cuda" --root ${root} \
									--output_txt ${file} \
									--SenMap_dir ${sens_dir} \
									--JPEG_enable True \
									--resize_compress True \
									--Qmax_Y 100 --Qmax_C 100 \
									--QF_Y ${qf} --QF_C ${qf} 
done


