g++ -std=c++11 main.cpp -o main_output $(pkg-config opencv4 --cflags --libs) -lpthread 

./main_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 62 -q 62 -B 1 -L 0

# ILSVRC2012_val_00017916.JPEG  lena3.tif
