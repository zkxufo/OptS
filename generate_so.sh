g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SDQ_module.cpp -o SDQ.so