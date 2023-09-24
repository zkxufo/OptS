#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./HDQ/SWE_OptS_QF_fixed.h"
using namespace std;

// -------------
// pure C++ code
// -------------
using namespace std;

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

std::tuple<py::array, py::array, float>  py__call__(py::array_t<float, py::array::c_style | py::array::forcecast> array,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> SenMap,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> ColorSpaceW_,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> InvColorSpaceW_,
                                       bool BiasPerImageFlag,
                                       int J, int a, int b, int QF_Y, int QF_C,
                                       float DT_Y, float DT_C, float d_waterlevel_Y, float d_waterlevel_C, 
                                       int Qmax_Y, int Qmax_C
                                       ){
  unsigned long size[2];
  size[0] = (unsigned long)array.shape()[1];
  size[1] = (unsigned long)array.shape()[2];
  vector<float> pos(array.size());
  vector<vector<vector<float>>> Vect_img(3, vector<vector<float>>(size[0], vector<float>(size[1], 0)));
  memcpy(pos.data(), array.data(),array.size()*sizeof(float));
  
  float ColorSpaceW[3][3];
  float InvColorSpaceW[3][3];

  memcpy(ColorSpaceW, ColorSpaceW_.data(), 3*3*sizeof(float));
  memcpy(InvColorSpaceW, InvColorSpaceW_.data(),3*3*sizeof(float));
  
  float BPP =0;
  vector<float> result(array.size());
  seq2img(pos, Vect_img, size[0], size[1]);
  
  float Sen_Map[3][64]={0};
  memcpy(Sen_Map, SenMap.data(), 3*64*sizeof(float));
  
  vector<vector<float>> Q_table(2, vector<float>(64, 0));
  
  unsigned long size_q = 64;
  vector<float> q_table(2 * size_q);

  float ImageBias[3] = {0,0,0};

  ColorSpaceConv(Vect_img, ColorSpaceW, ImageBias, BiasPerImageFlag);


  HDQ_OptD hdq;
  hdq.__init__(Sen_Map, QF_Y , QF_C, J, a, b, DT_Y, DT_C, d_waterlevel_Y, d_waterlevel_C, Qmax_Y, Qmax_C);
  BPP = hdq.__call__(Vect_img, q_table); // Vect_img is the compressed dequantilzed image after hdq.__call__()

  InvColorSpaceConv(Vect_img, InvColorSpaceW, ImageBias);

  int ndim_q = 2;
  vector<unsigned long> shape_q   = { 2, size_q};
  vector<unsigned long> strides_q = { size_q*sizeof(float), sizeof(float)};

  img2seq(Vect_img, result, size[0], size[1]);
  int ndim = 3;
  vector<unsigned long> shape   = { 3, size[0], size[1]};
  vector<unsigned long> strides = { size[0]*size[1]*sizeof(float),
                                    size[1]*sizeof(float), sizeof(float)};

  return std::make_tuple(
    py::array(py::buffer_info(
    result.data(),                          /* data as contiguous array */
    sizeof(float),                          /* size of one scalar       */
    py::format_descriptor<float>::format(), /* data type                */
    ndim,                                    /* number of dimensions     */
    shape,                                   /* shape of the matrix      */
    strides                                  /* strides for each axis    */
  )),
  py::array(py::buffer_info(
    q_table.data(),                          /* data as contiguous array */
    sizeof(float),                          /* size of one scalar       */
    py::format_descriptor<float>::format(), /* data type                */
    ndim_q,                                    /* number of dimensions     */
    shape_q,                                   /* shape of the matrix      */
    strides_q                                  /* strides for each axis    */
  )),
  BPP);
}
// wrap as Python module
PYBIND11_MODULE(SWE_OptS_QF_fixed,m)
{
  m.doc() = "SWE_OptS_QF_fixed API";
  m.def("__call__", &py__call__, py::return_value_policy::move ,"Calculate the length of an array of vectors");
}
