#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/highgui.hpp>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./SDQ/SDQ.h"
#include "./SDQ/load.h"
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
std::pair<py::array, float> py__call__(py::array_t<float, py::array::c_style | py::array::forcecast> array,
                     string Model, int J, int a, int b, int QF_Y, int QF_C, float Beta, float Lmbd, float eps){

  unsigned long size[2];
  size[0] = (unsigned long)array.shape()[1];
  size[1] = (unsigned long)array.shape()[2];
  // allocate std::vector (to pass to the C++ function)
  vector<float> pos(array.size());
  vector<vector<vector<float>>> Vect_img(3, vector<vector<float>>(size[0], vector<float>(size[1], 0)));
  // copy py::array -> std::vector
  memcpy(pos.data(), array.data(),array.size()*sizeof(float));
  //delete [] &array;
  // call pure C++ function
  //TODO::
  float BPP =0;
  vector<float> result(array.size());
  seq2img(pos, Vect_img, size[0], size[1]);
  float Lmbda = Lmbd;
  float Beta_S = Beta;
  float Beta_W = Beta;
  float Beta_X = Beta;  
  float Sen_Map[3][64]={0};
  LoadSenMap(Model, Sen_Map);
  float W_rgb2swx[3][3];
  float W_swx2rgb[3][3];
  LoadColorConvW(Model, W_rgb2swx, W_swx2rgb);
  float bias_rgb2swx = 128;
  
  // Mat2Vector(image, Vect_img);
  if(Model=="NoModel"){
    rgb2YUV(Vect_img, W_rgb2swx, bias_rgb2swx);
  }
  else{
    rgb2swx(Vect_img, W_rgb2swx, bias_rgb2swx);
  }
  SDQ sdq;
  sdq.__init__(eps, Beta_S, Beta_W, Beta_X, Lmbda, Sen_Map, QF_Y, QF_C, J, a ,b);
  BPP = sdq.__call__(Vect_img); // Vect_img is the compressed dequantilzed image after sdq.__call__()
  if(Model=="NoModel"){
    YUV2rgb(Vect_img, W_rgb2swx, bias_rgb2swx);
  }
  else{
    swx2rgb(Vect_img, W_swx2rgb, bias_rgb2swx);
  }
  img2seq(Vect_img, result, size[0], size[1]);
  int ndim = 3;
  vector<unsigned long> shape   = { 3, size[0], size[1]};
  vector<unsigned long> strides = { size[0]*size[1]*sizeof(float),
                                    size[1]*sizeof(float), sizeof(float)};
  // delete [] Sen_Map;
  // return 2-D NumPy array
  return std::make_pair(py::array(py::buffer_info(
    result.data(),                           /* data as contiguous array */
    sizeof(float),                          /* size of one scalar       */
    py::format_descriptor<float>::format(), /* data type                */
    ndim,                                    /* number of dimensions     */
    shape,                                   /* shape of the matrix      */
    strides                                  /* strides for each axis    */
  )), BPP);
}
// wrap as Python module
PYBIND11_MODULE(SDQ,m)
{
  m.doc() = "SDQ API";
  m.def("__call__", &py__call__, py::return_value_policy::move ,"Calculate the length of an array of vectors");
}
