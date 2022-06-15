g++ -std=c++11 SDQmain.cpp -o SDQoutput $(pkg-config opencv4 --cflags --libs) -lpthread 

# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 14.5 -q 50 -B 1 -L 0   # done
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 40 -q 50 -B 1 -L 0 # done
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 64.5 -q 50 -B 1 -L 0   # done
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 76.9 -q 50 -B 1 -L 0   # done

./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 13 -q 50 -B 1 -L 15  # done
./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 38 -q 50 -B 1 -L 15 # done
./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 65 -q 50 -B 1 -L 17 # done
./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 78 -q 50 -B 1 -L 8 # done
# ILSVRC2012_val_00017916.JPEG  lena3.tif
