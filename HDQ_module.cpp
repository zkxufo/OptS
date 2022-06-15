#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/highgui.hpp>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./HDQ/HDQ.h"
using namespace cv;
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
std::pair<py::array, double> py__call__(py::array_t<double, py::array::c_style | py::array::forcecast> array,
                     string Model, int J, int a, int b, int QF_Y, int QF_C, double Beta, double Lmbd, double eps){
  unsigned long size[2];
  size[0] = (unsigned long)array.shape()[1];
  size[1] = (unsigned long)array.shape()[2];
  // allocate std::vector (to pass to the C++ function)
  vector<double> pos(array.size());
  vector<vector<vector<double>>> Vect_img(3, vector<vector<double>>(size[0], vector<double>(size[1], 0)));
  // copy py::array -> std::vector
  memcpy(pos.data(), array.data(),array.size()*sizeof(double));
  // call pure C++ function
  //TODO::
  double BPP =0;
  vector<double> result(array.size());
  seq2img(pos, Vect_img, size[0], size[1]);
  double Lmbda = Lmbd;
  double Beta_S = Beta;
  double Beta_W = Beta;
  double Beta_X = Beta;  
  double W_rgb2swx[3][3];
  double W_swx2rgb[3][3];
  double bias_rgb2swx = 128;
  
  // Mat2Vector(image, Vect_img);
  rgb2YUV(Vect_img, W_rgb2swx, bias_rgb2swx);
  HDQ hdq;
  hdq.__init__(eps, Beta_S, Beta_W, Beta_X, Lmbda, QF_Y, QF_C, J, a ,b);
  BPP = hdq.__call__(Vect_img); // Vect_img is the compressed dequantilzed image after sdq.__call__()
  YUV2rgb(Vect_img, W_rgb2swx, bias_rgb2swx);
  img2seq(Vect_img, result, size[0], size[1]);
  int ndim = 3;
  vector<unsigned long> shape   = { 3, size[0], size[1]};
  vector<unsigned long> strides = { size[0]*size[1]*sizeof(double),
                                    size[1]*sizeof(double), sizeof(double)};
  // delete [] Sen_Map;
  // return 2-D NumPy array
  return std::make_pair(py::array(py::buffer_info(
    result.data(),                           /* data as contiguous array */
    sizeof(double),                          /* size of one scalar       */
    py::format_descriptor<double>::format(), /* data type                */
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
