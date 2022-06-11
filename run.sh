g++ -std=c++11 main.cpp -o main_output $(pkg-config opencv4 --cflags --libs) -lpthread 

./main_output -M NoModel -P ./sample/lena3.tif -J 4 -a 4 -b 4 -Q 10 -q 10 -B 1 -L 50

# ILSVRC2012_val_00017916.JPEG  lena3.tif