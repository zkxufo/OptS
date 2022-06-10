#include <math.h>
#include <iostream>
#include <vector>
using namespace std;
const double WR = 0.299;
const double WG = 0.587;
const double WB = 0.114;
void rgb2swx(std::vector<std::vector<std::vector<double>>>& rgb_img, double W_matrix[3][3], double bias){
    int i, j, k;
    double r_ch, g_ch, b_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    double ch[3];
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

void swx2rgb(std::vector<std::vector<std::vector<double>>>& rgb_img, double W_matrix[3][3], double bias){
    int i, j, k;
    double r_ch, g_ch, b_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    for(j=0; j<nrows; j++){
        for(k=0; k<ncols; k++){
            r_ch=rgb_img[0][j][k];
            g_ch=rgb_img[1][j][k];
            b_ch=rgb_img[2][j][k];
            rgb_img[0][j][k] = std::min(std::max(r_ch*W_matrix[0][0]+g_ch*W_matrix[0][1]+b_ch*W_matrix[0][2]+bias, 0.), 255.);
            rgb_img[1][j][k] = std::min(std::max(r_ch*W_matrix[1][0]+g_ch*W_matrix[1][1]+b_ch*W_matrix[1][2]+bias, 0.), 255.);
            rgb_img[2][j][k] = std::min(std::max(r_ch*W_matrix[2][0]+g_ch*W_matrix[2][1]+b_ch*W_matrix[2][2]+bias, 0.), 255.);
        }
    }
}

void rgb2YUV(std::vector<std::vector<std::vector<double>>>& rgb_img, double W_matrix[3][3], double bias){
    int i, j, k;
    double r_ch, g_ch, b_ch;
    double Y_ch, U_ch, V_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    double ch[3];
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

void YUV2rgb(std::vector<std::vector<std::vector<double>>>& rgb_img, double W_matrix[3][3], double bias){
    int i, j, k;
    double Y_ch, U_ch, V_ch;
    double r_ch, g_ch, b_ch;
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
            rgb_img[0][j][k] = min(max(r_ch, 0.), 255.);
            rgb_img[1][j][k] = min(max(g_ch, 0.), 255.);
            rgb_img[2][j][k] = min(max(b_ch, 0.), 255.);
        }
    }
}
