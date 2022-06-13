g++ -std=c++11 main.cpp -o main_output $(pkg-config opencv4 --cflags --libs) -lpthread 

./main_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 14.5 -q 50 -B 1 -L 0
./main_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 40 -q 50 -B 1 -L 0
./main_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 64.5 -q 50 -B 1 -L 0
./main_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 76.9 -q 50 -B 1 -L 0
# ILSVRC2012_val_00017916.JPEG  lena3.tif
