// Blockify.h

// MIT License

// Copyright (c) 2022 deponce(Linfeng Ye), University of Waterloo

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <math.h>
#include <iostream>
using namespace std;

int pad_shape(int Num, int size=8){
    /*
    :Fn  pad_shape: calculate the padded shape
    :param Num: number of row or col of image 
    :param size: block size
    :return int: padded size
    */
    int res =  Num%size;
    int pad = 1;
    if (res == 0){
        pad = 0;
        }
    int n = (Num/size+pad)*size;
    return n;
}

 void Mat2Vector(cv::Mat Mat_img, vector<vector<vector<float>>>& Vect_img){
    /*
    :Fn  Mat2Vector: 
    :param cv::Mat Mat_img:
    :param vector<vector<vector<float>>>& Vect_img:
    :return void:
    */
     int i, j;
     int nrows = Mat_img.rows;
     int ncols = Mat_img.cols;
     for (i=0; i<nrows; ++i){
         for(j=0; j<ncols; ++j){
             Vect_img[0][i][j] =  Mat_img.at<cv::Vec3b>(i, j)[2];
             Vect_img[1][i][j] =  Mat_img.at<cv::Vec3b>(i, j)[1];
             Vect_img[2][i][j] =  Mat_img.at<cv::Vec3b>(i, j)[0];
         }
     }
 }

 void Vector2Mat(vector<vector<vector<float>>> Vect_img, cv::Mat Mat_img){
     int i, j;
     int nrows = Mat_img.rows;
     int ncols = Mat_img.cols;
     for (i=0; i<nrows; ++i){
         for(j=0; j<ncols; ++j){
             Mat_img.at<cv::Vec3b>(i, j)[2] = Vect_img[0][i][j];
             Mat_img.at<cv::Vec3b>(i, j)[1] = Vect_img[1][i][j];
             Mat_img.at<cv::Vec3b>(i, j)[0] = Vect_img[2][i][j];
         }
     }
 }

void blockify(vector<vector<float>> img, int img_size[2], float v_im[][8][8]){
    int pad_row = pad_shape(img_size[0])/8; 
    int pad_col = pad_shape(img_size[1])/8; 
    int N;
    int x_idx, y_idx, i,j,w,h;
    for (i=0; i<pad_row; ++i){
        for (j=0; j<pad_col; ++j){
            N = i*pad_col+j;
            for(w=0; w<8; ++w){
                for(h=0; h<8; ++h){
                    x_idx = i*8+w;
                    y_idx = j*8+h;
                    if (x_idx<img_size[0] && y_idx<img_size[1]){
                        v_im[N][w][h] = img[x_idx][y_idx]; 
                    }
                    else{
                        v_im[N][w][h] = 0;
                    }
                }
            }
        }
    }
}

void deblockify(float blockified_img[][8][8], vector<vector<float>>& deblockify_image, int img_size[2]){
    int pad_row = pad_shape(img_size[0])/8;
    int pad_col = pad_shape(img_size[1])/8;
    int N = 0;
    int x_idx, y_idx;
    int n_blocks = pad_col*pad_row/64;
    float pix_val;
    for(int i=0; i<pad_row; ++i){
        for(int j=0; j<pad_col; ++j){
            N = i*pad_col+j;
            for(int w=0; w<8; ++w){
                for(int h=0; h<8; ++h){
                    x_idx = i*8+w;
                    y_idx = j*8+h;
                    if(x_idx<img_size[0] && y_idx<img_size[1]){
                        pix_val = blockified_img[N][w][h];
                        deblockify_image[x_idx][y_idx]= pix_val;
                    }
                }
            }
        }
    }
}

vector<vector<vector<float>>> pad_Vector(vector<vector<vector<float>>> Vector_img){
    int nrow = Vector_img[0].size();
    int ncol = Vector_img[0][0].size();
    int pad_row = pad_shape(nrow);
    int pad_col = pad_shape(ncol);
    vector<vector<vector<float>>> pad_img(3, vector<vector<float>>(pad_row, vector<float>(pad_col, 0)));
    int pix_val;
    for(int c=0; c<3;c++){
        for(int i=0; i<pad_row; ++i){
            for(int j=0; j<pad_col; ++j){
                if(i<nrow && j<ncol){
                    pad_img[c][i][j] = Vector_img[c][i][j];
                }
            }
        }
    }
    return pad_img;
}

