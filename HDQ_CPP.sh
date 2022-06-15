g++ -std=c++11 HDQmain.cpp -o HDQ_output $(pkg-config opencv4 --cflags --libs) -lpthread 
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 14 -q 50 -B 1 -L 0   # done
echo
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 15 -q 50 -B 1 -L 0   # done
echo
echo
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 39 -q 50 -B 1 -L 15 # done
echo
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 40.5 -q 50 -B 1 -L 15 # done
echo
echo
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 63 -q 50 -B 1 -L 0   # done
echo
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 64 -q 50 -B 1 -L 0   # done
echo
echo
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 76 -q 50 -B 1 -L 0   # done
echo
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 77 -q 50 -B 1 -L 0   # done
echo
