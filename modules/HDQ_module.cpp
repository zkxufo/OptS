#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./HDQ/HDQ.h"

using namespace std;

// -------------
// pure C++ code
// -------------
using namespace std;

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
std::pair<py::array, float> py__call__(py::array_t<float, py::array::c_style | py::array::forcecast> array,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> ColorSpaceW_,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> InvColorSpaceW_,
                                       bool BiasPerImageFlag,
                                       int J, int a, int b, int QF_Y, int QF_C){
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

  float BPP = 0;
  vector<float> result(array.size());
  seq2img(pos, Vect_img, size[0], size[1]);
  float ImageBias[3] = {0,0,0};


  ColorSpaceConv(Vect_img, ColorSpaceW, ImageBias, BiasPerImageFlag);

  HDQ hdq;
  hdq.__init__(QF_Y, QF_C, J, a ,b);
  BPP = hdq.__call__(Vect_img); // Vect_img is the compressed dequantilzed image after hdq.__call__()

  InvColorSpaceConv(Vect_img, InvColorSpaceW, ImageBias);

  img2seq(Vect_img, result, size[0], size[1]);
  int ndim = 3;
  vector<unsigned long> shape   = { 3, size[0], size[1]};
  vector<unsigned long> strides = { size[0]*size[1]*sizeof(float),
                                    size[1]*sizeof(float), sizeof(float)};

  return std::make_pair(py::array(py::buffer_info(
    result.data(),                          /* data as contiguous array */
    sizeof(float),                          /* size of one scalar       */
    py::format_descriptor<float>::format(), /* data type                */
    ndim,                                    /* number of dimensions     */
    shape,                                   /* shape of the matrix      */
    strides                                  /* strides for each axis    */
  )), BPP);
}
// wrap as Python module
PYBIND11_MODULE(HDQ,m)
{
  m.doc() = "HDQ API";
  m.def("__call__", &py__call__, py::return_value_policy::move ,"Calculate the length of an array of vectors");
}
