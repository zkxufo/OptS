GPU_ID=2

# python3 get_DCTgrad.py -model Resnet18 -Batch_size 100 -Nexample 100000
# python3 get_DCTgrad.py -model Resnet34 -Batch_size 50 -Nexample 100000
# python3 get_DCTgrad.py -model Resnet101 -Batch_size 50 -Nexample 100000
# python3 get_DCTgrad.py -model Resnet50 -Batch_size 50 -Nexample 100000 
# python3 get_DCTgrad.py -model Resnet152 -Batch_size 20 -Nexample 100000 

# python3 get_DCTgrad.py -model convnext_base -Batch_size 10 -Nexample 100000
# python3 get_DCTgrad.py -model convnext_tiny -Batch_size 10 -Nexample 100000
# python3 get_DCTgrad.py -model convnext_large -Batch_size 2 -Nexample 100000
# python3 get_DCTgrad.py -model convnext_small -Batch_size 10 -Nexample 100000 

# CUDA_VISIBLE_DEVICES=${GPU_ID} python3 get_DCTgrad.py -model Regnet800mf -Batch_size 10 -Nexample 10000
# CUDA_VISIBLE_DEVICES=${GPU_ID} python3 get_DCTgrad.py -model Regnet6gf -Batch_size 10 -Nexample 10000
# CUDA_VISIBLE_DEVICES=${GPU_ID} python3 get_DCTgrad.py -model regnet2gf -Batch_size 10 -Nexample 10000
# CUDA_VISIBLE_DEVICES=${GPU_ID} python3 get_DCTgrad.py -model regnet8gf -Batch_size 10 -Nexample 10000
# CUDA_VISIBLE_DEVICES=${GPU_ID} python3 get_DCTgrad.py -model regnet16gf -Batch_size 10 -Nexample 10000
# CUDA_VISIBLE_DEVICES=${GPU_ID} python3 get_DCTgrad.py -model regnet32gf -Batch_size 10 -Nexample 10000

#  --------------------------------------------------------------------------------------------------


python3 plot.py -model Regnet800mf
python3 plot.py -model Regnet6gf
python3 plot.py -model regnet2gf
python3 plot.py -model regnet8gf
python3 plot.py -model regnet16gf
python3 plot.py -model regnet32gf


# python3 plot.py -model Resnet18
# python3 plot.py -model Resnet34
# python3 plot.py -model Resnet101
# python3 plot.py -model Resnet50
# python3 plot.py -model Resnet152

# python3 plot.py -model convnext_base
# python3 plot.py -model convnext_tiny
# python3 plot.py -model convnext_small
# python3 plot.py -model convnext_large




