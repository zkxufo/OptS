# g++ -c -o ./jpeg_standard/*.o ./jpeg_standard/*.cpp -I./include -std=c++11

g++ -std=c++11 *.cpp -o main_output $(pkg-config opencv4 --cflags --libs) -lpthread 

./main_output -M NoModel -P ./sample/lena3.tif -J 4 -a 4 -b 4 -Q 10 -q 10 -B 1 -L 0.000001

# ILSVRC2012_val_00017916.JPEG  lena3.tif
# g++ *.cpp -o main -lpthread -D_GLIBCXX_USE_CXX11_ABI=0

# current_jpeg=/home/h2amer/AhmedH.Salamah/workspace_pc15/JPEG_cpp/SDQ-based-codec-fro-DNNs-perception/jpeg_customized_latest/sample.jpeg
# output_path_to_files=/home/h2amer/AhmedH.Salamah/workspace_pc15/JPEG_cpp/SDQ-based-codec-fro-DNNs-perception/jpeg_customized_latest/
# output_txt=/home/h2amer/AhmedH.Salamah/workspace_pc15/JPEG_cpp/SDQ-based-codec-fro-DNNs-perception/jpeg_customized_latest/sample/

# ./main $current_jpeg $output_path_to_files $output_txt
