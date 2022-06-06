cmake_minimum_required(VERSION 2.8.12)
project(SDQ)

find_package( PkgConfig )
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_subdirectory(pybind11)
pybind11_add_module(SDQ SDQ_module.cpp)
