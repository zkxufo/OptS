#include <math.h>
#include <iostream>
#include "DCT.h"
#include "ColorSpaceConv.h"
#include "Quantize.h"
#include "Blockify.h"
using namespace std;
// using namespace cv;
//TODO:: check N and ID is necessary

const double MAX_PXL_VAL = 255.;

struct node{
    double cost[64];
    int ID[64]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int rs[64]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double ent = 0;
};

double sign(double C){
    double Sign;
    if (C >= 0){
        Sign = 1;
    }
    if( C < 0){
        Sign = -1;
    }
    return Sign;
}

int argmin(double J_lst[63]){
    //return the index of min entry in the list
    int idx=0;
    int i;
    double min_ent = J_lst[0];
    for(i=0; i<63; i++){
        if (J_lst[i]<min_ent){
            min_ent = J_lst[i];
            idx = i;
        }
    }
    return idx;
}

void shape(vector<vector<vector<double>>> img, int S[2]){
    S[0] = img[0].size();
    S[1] = img[0][0].size();
}

int size_group(double num, const int MAX_val=10, const int MIN_val = 0){
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

void cumsum(double C[64], double CumC[64], double Sen_Map[64]){
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

int find_the_last_non_zero(double arr[64]){
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

double MSE3C(vector<vector<vector<double>>>img1,
           vector<vector<vector<double>>>img2){
    double m = 0;
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

double PSNR3C(vector<vector<vector<double>>> img1,
           vector<vector<vector<double>>> img2){
    double m = MSE3C(img1, img2);
    return 10*log10(pow(MAX_PXL_VAL,2)/m);
}

double MSEY(vector<vector<vector<double>>> img1,
           vector<vector<vector<double>>> img2){
    double m = 0;
    double Y1, Y2;
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

double PSNRY(vector<vector<vector<double>>> img1,
           vector<vector<vector<double>>> img2){
    double m = MSEY(img1, img2);
    cout<<"m: "<<m<<endl;
    return 10*log10(pow(MAX_PXL_VAL,2)/m);
}
