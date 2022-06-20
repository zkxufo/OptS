// ColorSpaceConv.h

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
#include <vector>
using namespace std;
const float WR = 0.299;
const float WG = 0.587;
const float WB = 0.114;
const float MAX_PXL_VAL = 255.;
const float MIN_PXL_VAL = 0.;
void rgb2swx(std::vector<std::vector<std::vector<float>>>& rgb_img, float W_matrix[3][3], float bias){
    int i, j, k;
    float r_ch, g_ch, b_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    float ch[3];
    for(i=0; i<nrows; i++){
        for(j=0; j<ncols; j++){
            r_ch = rgb_img[0][i][j]-bias;
            g_ch = rgb_img[1][i][j]-bias;
            b_ch = rgb_img[2][i][j]-bias;
            rgb_img[0][i][j] = r_ch*W_matrix[0][0]+g_ch*W_matrix[0][1]+b_ch*W_matrix[0][2];
            rgb_img[1][i][j] = r_ch*W_matrix[1][0]+g_ch*W_matrix[1][1]+b_ch*W_matrix[1][2];
            rgb_img[2][i][j] = r_ch*W_matrix[2][0]+g_ch*W_matrix[2][1]+b_ch*W_matrix[2][2];
        }
    }
}

void swx2rgb(std::vector<std::vector<std::vector<float>>>& rgb_img, float W_matrix[3][3], float bias){
    int i, j, k;
    float r_ch, g_ch, b_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    for(j=0; j<nrows; j++){
        for(k=0; k<ncols; k++){
            r_ch=rgb_img[0][j][k];
            g_ch=rgb_img[1][j][k];
            b_ch=rgb_img[2][j][k];
            rgb_img[0][j][k] = std::min(std::max(r_ch*W_matrix[0][0]+g_ch*W_matrix[0][1]+b_ch*W_matrix[0][2]+bias, MIN_PXL_VAL), MAX_PXL_VAL);
            rgb_img[1][j][k] = std::min(std::max(r_ch*W_matrix[1][0]+g_ch*W_matrix[1][1]+b_ch*W_matrix[1][2]+bias, MIN_PXL_VAL), MAX_PXL_VAL);
            rgb_img[2][j][k] = std::min(std::max(r_ch*W_matrix[2][0]+g_ch*W_matrix[2][1]+b_ch*W_matrix[2][2]+bias, MIN_PXL_VAL), MAX_PXL_VAL);
        }
    }
}

void rgb2YUV(std::vector<std::vector<std::vector<float>>>& rgb_img, float W_matrix[3][3], float bias){
    int i, j, k;
    float r_ch, g_ch, b_ch;
    float Y_ch, U_ch, V_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    float ch[3];
    for(i=0; i<nrows; i++){
        for(j=0; j<ncols; j++){
            r_ch = rgb_img[0][i][j];
            g_ch = rgb_img[1][i][j];
            b_ch = rgb_img[2][i][j];
            Y_ch = r_ch*WR+g_ch*WG+b_ch*WB;
            U_ch = (b_ch-Y_ch)/(1-WB)/2;
            V_ch = (r_ch-Y_ch)/(1-WR)/2;
            rgb_img[0][i][j] = Y_ch;
            rgb_img[1][i][j] = U_ch;
            rgb_img[2][i][j] = V_ch;
        }
    }
}

void YUV2rgb(std::vector<std::vector<std::vector<float>>>& rgb_img, float W_matrix[3][3], float bias){
    int i, j, k;
    float Y_ch, U_ch, V_ch;
    float r_ch, g_ch, b_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    for(j=0; j<nrows; j++){
        for(k=0; k<ncols; k++){
            Y_ch=rgb_img[0][j][k];
            U_ch=rgb_img[1][j][k];
            V_ch=rgb_img[2][j][k];
            r_ch = Y_ch+2*(1-WR)*V_ch;
            g_ch = Y_ch-2*(1-WB)*WB/WG*U_ch-2*(1-WR)*WR/WG*V_ch;
            b_ch = Y_ch+2*(1-WB)*U_ch;
            rgb_img[0][j][k] = min(max(r_ch, MIN_PXL_VAL), MAX_PXL_VAL);
            rgb_img[1][j][k] = min(max(g_ch, MIN_PXL_VAL), MAX_PXL_VAL);
            rgb_img[2][j][k] = min(max(b_ch, MIN_PXL_VAL), MAX_PXL_VAL);
        }
    }
}
