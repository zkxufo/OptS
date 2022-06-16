#include <math.h>
#include <iostream>
#include "DCT.h"
#include "ColorSpaceConv.h"
#include "Quantize.h"
#include "Blockify.h"
using namespace std;

const float FLAG_SIZE =  8*(1+1) // SOI
                        + 8*(1+1+2+5+1+1+2+2+1) // APP0
                        + 8*(1+1+2+1+1+64) // DQT
                        + 8*(1+1+2+1+2+2+1+1+1+1) // SOF0
                        // TODO: cal n in DHT and change 2 to 4
                        + 8*(1+1+2+1+16)*4+256*4 //DHT
                        + 8*(1+1+2+1+1+1+3) + 8*(2+2+1+2)// SOS
                        + 8*(1+1); //EOI;

struct node{
    float cost[64];
    int ID[64]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int rs[64]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    float ent = 0;
};

float sign(float C){
    float Sign;
    if (C >= 0){
        Sign = 1;
    }
    if( C < 0){
        Sign = -1;
    }
    return Sign;
}

int argmin(float J_lst[63]){
    //return the index of min entry in the list
    int idx=0;
    int i;
    float min_ent = J_lst[0];
    for(i=0; i<63; i++){
        if (J_lst[i]<min_ent){
            min_ent = J_lst[i];
            idx = i;
        }
    }
    return idx;
}

void shape(vector<vector<vector<float>>> img, int S[2]){
    S[0] = img[0].size();
    S[1] = img[0][0].size();
}

int size_group(float num, const int MAX_val=10, const int MIN_val = 0){
    int S;
    int abs_num = abs(num);
    switch (abs_num){
        case 0 :{S = 0;break;}
        case 1 :{S = 1;break;}
        case 2 ... 3:{S = 2;break;}
        case 4 ... 7:{S = 3;break;}
        case 8 ... 15:{S = 4;break;}
        case 16 ... 31:{S = 5;break;}
        case 32 ... 63:{S = 6;break;}
        case 64 ... 127:{S = 7;break;}
        case 128 ... 255:{S = 8;break;}
        case 256 ... 511:{S = 9;break;}
        case 512 ... 1023:{S = 10;break;}
        default:{S =11;break;}
    }
    return min(max(S,MIN_val), MAX_val);
}

void cumsum(float C[64], float CumC[64], float Sen_Map[64]){
    CumC[0] = 0;
    int i;
    for(i=1; i<64; i++){
        CumC[i] = CumC[i-1]+Sen_Map[i]*pow(C[i], 2);
    }
}

int rs2hash(int r, int s){
    //Run-Size pair -> hash value
    int hash_val = r<<4;
    hash_val += s;
    return hash_val;
}

void hash2rs(int hash_val, int & r, int & s){
    //hash value -> Run-Size pair
    r = hash_val>>4;
    s = hash_val%16;
}

int find_the_last_non_zero(float arr[64]){
    int i;
    int idx=0;
    for(i=63; i>0; i--){
        if(arr[i] ==0){
            idx += 1;
        }
        else{
            break;
        }
    }
    return 63-idx;
}

void seq2img(vector<float> pos, vector<vector<vector<float>>>& img, int nrow, int ncol){
  for(int c=0; c<3; ++c){
    for(int i=0; i<nrow; ++i){
      for(int j=0; j<ncol; ++j){
        img[c][i][j] = pos[c*nrow*ncol+i*ncol+j];
      }
    }
  }
}

void img2seq(vector<vector<vector<float>>> img, vector<float>& pos, int nrow, int ncol){
  for(int c=0; c<3; ++c){
    for(int i=0; i<nrow; ++i){
      for(int j=0; j<ncol; ++j){
        pos[c*nrow*ncol+i*ncol+j] = img[c][i][j];
      }
    }
  }
}

void Subsampling(vector<vector<float>>& img,int img_shape_Y[2],
                 int img_shape_C[2], int J, int a, int b){
    int Nrows = img_shape_Y[0];
    int Ncols = img_shape_Y[1];
    int pad_rows = pad_shape(Nrows, J);
    int pad_cols = pad_shape(Ncols, J);
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int i;
    int j;
    int hidx;
    int vidx;
    for(i=0, hidx=0; i<pad_rows; i+=SmplVstep, hidx++){
        for(j=0, vidx=0;j<pad_cols;j+=SmplHstep,vidx++){
            if(j<Ncols && i<Nrows){
                img[hidx][vidx] = img[i][j];
            }
        }
    }
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    img_shape_C[0] = min(Nrows,Smplrows);
    img_shape_C[1] = min(Ncols,Smplcols);
}

void Upsampling(vector<vector<float>>& img, int img_shape_Y[2], 
                int J, int a, int b){
    int Nsampled_row = img_shape_Y[0];
    int Nsampled_col = img_shape_Y[1];
    int x,y,xidx,yidx;
    int i,j;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    for(i=Nsampled_row-1;i>-1;i--){
        for(j=Nsampled_col-1;j>-1;j--){
            for(x=SmplHstep-1; x>-1; x--){
                for(y=SmplVstep-1; y>-1; y--){
                    yidx = i*SmplVstep+y;
                    xidx = j*SmplHstep+x;
                    if(xidx<Nsampled_col && yidx<Nsampled_row){
                        img[yidx][xidx] = img[i][j];
                    }
                }
            }
        }
    }
}


float MSE3C(vector<vector<vector<float>>>img1,
           vector<vector<vector<float>>>img2){
    float m = 0;
    int nrows = img1[0].size();
    int ncols = img1[0][0].size();
    for(int i=0; i<nrows; i++){
        for(int j=0; j<ncols; j++){
            m += pow(img1[0][j][i]-img2[0][j][i], 2);
            m += pow(img1[1][j][i]-img2[1][j][i], 2);
            m += pow(img1[2][j][i]-img2[2][j][i], 2);
        }
    }
    m = m/3/nrows/ncols;
    return m;
}

float PSNR3C(vector<vector<vector<float>>> img1,
           vector<vector<vector<float>>> img2){
    float m = MSE3C(img1, img2);
    return 10*log10(pow(MAX_PXL_VAL,2)/m);
}

float MSEY(vector<vector<vector<float>>> img1,
           vector<vector<vector<float>>> img2){
    float m = 0;
    float Y1, Y2;
    int nrows = img1[0].size();
    int ncols = img1[0][0].size();
    for(int i=0; i<nrows; i++){
        for(int j=0; j<ncols; j++){
            Y1 = 0.299*img1[0][i][j] + 0.587*img1[1][i][j] + 0.114*img1[2][i][j];
            Y2 = 0.299*img2[0][i][j] + 0.587*img2[1][i][j] + 0.114*img2[2][i][j];
            m += pow((Y1-Y2), 2);
        }
    }
    m = m/nrows/ncols;
    return m;
}

float PSNRY(vector<vector<vector<float>>> img1,
           vector<vector<vector<float>>> img2){
    float m = MSEY(img1, img2);
    return 10*log10(pow(MAX_PXL_VAL,2)/m);
}
