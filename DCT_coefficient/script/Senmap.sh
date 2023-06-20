# python3 get_DCTgrad.py -model Resnet18 -Batch_size 200 -Nexample 100000
# python3 plot.py -model Resnet18

python3 get_DCTgrad.py -model Resnet18 -Batch_size 200 -Nexample 100000
python3 plot.py -model Resnet34

# python3 get_DCTgrad.py -model Squeezenet -Batch_size 200 -Nexample 100000
# python3 plot.py -model Squeezenet

# python3 get_DCTgrad.py -model Shufflenetv2 -Batch_size 200 -Nexample 100000
# python3 plot.py -model Shufflenetv2

# python3 get_DCTgrad.py -model Regnet -Batch_size 200 -Nexample 100000
# python3 plot.py -model Regnet
# echo
# python3 get_DCTgrad.py -model Mnasnet -Batch_size 100 -Nexample 100000
# echo 
# python3 plot.py -model Mnasnet

# echo 
# python3 get_DCTgrad.py -model Alexnet -Batch_size 100 -Nexample 100000
# echo 
# python3 plot.py -model Alexnet
# echo 
# python3 get_DCTgrad.py -model mobilenet_v2 -Batch_size 100 -Nexample 100000
# echo 
# python3 plot.py -model mobilenet_v2
