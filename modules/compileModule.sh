g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 OptD_module.cpp -o ../SWEmatching/OptD.so
g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 HDQ_module.cpp -o ../SWEmatching/HDQ.so


g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SWE_OptD_QF_fixed_module.cpp -o ../SWEmatching/SWE_OptD_QF_fixed.so
g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SWE_OptS_QF_fixed_module.cpp -o ../SWEmatching/SWE_OptS_QF_fixed.so
g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SWE_JPEG_d_fixed_module.cpp -o ../SWEmatching/SWE_JPEG_d_fixed.so
g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SWE_OptD_d_fixed_module.cpp -o ../SWEmatching/SWE_OptD_d_fixed.so
