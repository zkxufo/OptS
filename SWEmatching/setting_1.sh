export root=/home/l44ye/DATASETS/
export model=$1
echo ${model}
export sens_dir=./SenMap/


# Remove it for normal inference accuracy --> resize_compress
python HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
								--batchsize 10 \
								--device "cuda" --root ${root} \
								--SenMap_dir ${sens_dir} \
								--OptS_enable True \
								--Qmax_Y 100 --Qmax_C 100 


# Fixing QF: OptS Matching JPEG 
for qf in `seq 98 -1 70`
do
	python3 HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
									--batchsize 50 \
									--device "cuda" --root ${root} \
									--SenMap_dir ${sens_dir} \
									--OptS_enable True \
									--Qmax_Y 100 --Qmax_C 100 \
									--QF_Y ${qf} --QF_C ${qf} \
									--resize_compress True 
									
done


# Fixing QF: OptD Matching JPEG 
for qf in `seq 98 -1 70`
do
	python3 HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
									--batchsize 50 \
									--device "cuda" --root ${root} \
									--SenMap_dir ${sens_dir} \
									--resize_compress True \
									--OptD_enable True \
									--Qmax_Y 100 --Qmax_C 100 \
									--QF_Y ${qf} --QF_C ${qf} 
done

# Normal JEPG
for qf in `seq 98 -1 70`
do
	python3 HDQ_matching.py --Model ${model} --J 4 --a 4 --b 4 \
									--batchsize 10 \
									--device "cuda" --root ${root} \
									--SenMap_dir ${sens_dir} \
									--JPEG_enable True \
									--resize_compress True \
									--Qmax_Y 100 --Qmax_C 100 \
									--QF_Y ${qf} --QF_C ${qf} 
done



