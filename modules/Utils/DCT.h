#include <math.h>
#include <iostream>


using namespace std;

const float BMA[3] = {-0.27589937928294306, -0.7856949583871022, 0.7653668647301795};
const float MAMB[3] = {-1.3870398453221475, -1.1758756024193586, -1.8477590650225737};
const float A[3] = {0.8314696123025452, 0.9807852804032304, 0.5411961001461971};

const float IA[3] = {0.541196100146197, 0.8314696123025452, 0.9807852804032304};
const float IBMA[3] = {-1.8477590650225735, -1.3870398453221475, -1.1758756024193586};
const float IMAMB[3] = {0.7653668647301793, -0.27589937928294306, -0.7856949583871022};

const int ZIGZAG[8][8] = {{0,  1,  5,  6,  14, 15, 27, 28},

                          {2,  4,  7,  13, 16, 26, 29, 42},

                          {3,  8,  12, 17, 25, 30, 41, 43},

                          {9,  11, 18, 24, 31, 40, 44, 53},

                          {10, 19, 23, 32, 39, 45, 52, 54},

                          {20, 22, 33, 38, 46, 51, 55, 60},

                          {21, 34, 37, 47, 50, 56, 59, 61},

                          {35, 36, 48, 49, 57, 58, 62, 63}};

void rotation(float *s1, int idx1, int idx2,
              float *s2, int k){
    float tmp = A[k]*(s1[idx1]+s1[idx2]);
    s2[idx1] = BMA[k]*(s1[idx2]) + tmp;
    s2[idx2] = MAMB[k]*(s1[idx1]) + tmp;
}

void D1DCT(float a[8], float s2[8]){
    //tested
    float s1[8];
    //stage 1
    s1[0] = a[0] + a[7]; s1[1] = a[1] + a[6]; s1[2] = a[2] + a[5]; s1[3] = a[3] + a[4];
    s1[4] = a[3] - a[4]; s1[5] = a[2] - a[5]; s1[6] = a[1] - a[6]; s1[7] = a[0] - a[7];
    //stage 2
    s2[0] = s1[0] + s1[3]; s2[1] = s1[1] + s1[2]; s2[2] = s1[1] - s1[2];
    s2[3] = s1[0] - s1[3]; rotation(s1,4,7,s2,0); rotation(s1,5,6,s2,1);
    //stage 3
    s1[0] = s2[0] + s2[1]; s1[1] = s2[0] - s2[1]; rotation(s2,2,3,s1,2); s1[4] = s2[4] + s2[6];
    s1[5] = s2[7] - s2[5]; s1[6] = s2[4] - s2[6]; s1[7] = s2[5] + s2[7];
    //stage 4
    s2[0] = s1[0]; s2[4] = s1[1]; s2[2] = s1[2]; s2[6] = s1[3]; s2[7] = (s1[7] - s1[4]);
    s2[3] = s1[5]/2*sqrt(8); s2[5] = s1[6]/2*sqrt(8); s2[1] = (s1[7] + s1[4]);
}

void block_2_seqdct(float blockified_img_Y[][8][8],
                    float seq_dct_coefs_Y[][64],
                    int N_block){
    //tested
    float tmp_Y[8];
    float res_Y[8][8];
    int idx, N, i, j;
    for (N=0; N<N_block; N++){
        for (i=0; i<8; i++){
            D1DCT(blockified_img_Y[N][i], tmp_Y);
            for (j=0; j<8; j++){
                res_Y[j][i] = tmp_Y[j];
            }
        }  
        for (i=0; i<8; i++){
            D1DCT(res_Y[i], tmp_Y);
            for (j=0; j<8; j++){
                idx = ZIGZAG[j][i];
                seq_dct_coefs_Y[N][idx] = tmp_Y[j]/8.;
            }
        }
    }
}


void deepcopy(float X[][64], float Y[][64], int N_block)
{

    for(int N=0; N<N_block; N++){
        for (int idx = 0 ; idx<64 ; idx++)
        {
            Y[N][idx] = X[N][idx];
        }
    }

}

void irotation(float *s1, int idx1, int idx2,

               float *s2, int k){

    float tmp = IA[k]*(s1[idx1]+s1[idx2]);

    s2[idx1] = IBMA[k]*(s1[idx2]) + tmp;

    s2[idx2] = IMAMB[k]*(s1[idx1]) + tmp;

}

 

void D1IDCT(float a[8], float s2[8]){ 

    float s1[8];

    // stage 1

    s1[0] = a[0]; s1[1] = a[4]; s1[2] = a[2]; s1[3] = a[6]; s1[4] = a[1]-a[7];

    s1[5] = a[3]*sqrt(2); s1[6] = a[5]*sqrt(2); s1[7] = a[1]+a[7];

    // stage 2

    s2[0] = s1[0] + s1[1]; s2[1] = s1[0] - s1[1]; irotation(s1,2,3,s2,0); s2[4] = s1[4] + s1[6];

    s2[5] = s1[7] - s1[5]; s2[6] = s1[4] - s1[6]; s2[7] = s1[5] + s1[7];

    // stage 3

    s1[0] = s2[0] + s2[3]; s1[1] = s2[1] + s2[2]; s1[2] = s2[1] - s2[2];

    s1[3] = s2[0] - s2[3]; irotation(s2,4,7,s1,1); irotation(s2,5,6,s1,2);

    // stage 4

    s2[0] = s1[0] + s1[7]; s2[1] = s1[1] + s1[6]; s2[2] = s1[2] + s1[5]; s2[3] = s1[3] + s1[4];

    s2[4] = s1[3] - s1[4]; s2[5] = s1[2] - s1[5]; s2[6] = s1[1] - s1[6]; s2[7] = s1[0] - s1[7];

}

 

void D1IDCT(float a[64], int i, float s2[8]){ 

    float s1[8];

    // stage 1

    s1[0] = a[ZIGZAG[i][0]]; s1[1] = a[ZIGZAG[i][4]]; s1[2] = a[ZIGZAG[i][2]]; s1[3] = a[ZIGZAG[i][6]];

    s1[4] = a[ZIGZAG[i][1]]-a[ZIGZAG[i][7]]; s1[5] = a[ZIGZAG[i][3]]*sqrt(2);

    s1[6] = a[ZIGZAG[i][5]]*sqrt(2); s1[7] = a[ZIGZAG[i][1]]+a[ZIGZAG[i][7]];

    // stage 2

    s2[0] = s1[0] + s1[1]; s2[1] = s1[0] - s1[1]; irotation(s1,2,3,s2,0); s2[4] = s1[4] + s1[6];

    s2[5] = s1[7] - s1[5]; s2[6] = s1[4] - s1[6]; s2[7] = s1[5] + s1[7];

    s1[0] = s2[0] + s2[3]; s1[1] = s2[1] + s2[2]; s1[2] = s2[1] - s2[2];

    s1[3] = s2[0] - s2[3]; irotation(s2,4,7,s1,1); irotation(s2,5,6,s1,2);

    // stage 4

    s2[0] = s1[0] + s1[7]; s2[1] = s1[1] + s1[6]; s2[2] = s1[2] + s1[5]; s2[3] = s1[3] + s1[4];

    s2[4] = s1[3] - s1[4]; s2[5] = s1[2] - s1[5]; s2[6] = s1[1] - s1[6]; s2[7] = s1[0] - s1[7];

}


void seq_2_blockidct(float seq_dct_coefs_Y[][64], float blockified_img_Y[][8][8], int N_block){
    float tmp_Y[8];
    float res_Y[8][8];
    int idx, c, N, i, j;
    for (N=0; N<N_block; N++){
        for (i=0; i<8; i++){
            D1IDCT(seq_dct_coefs_Y[N], i, tmp_Y);
            for (j=0; j<8; j++){
                res_Y[j][i] = tmp_Y[j];
            }
        }
        for (i=0; i<8; i++){
            D1IDCT(res_Y[i], tmp_Y);
            for (j=0; j<8; j++){
                blockified_img_Y[N][j][i] = tmp_Y[j]/8.;
            }
        }
    }
}

void seq_2_block(float seq_dct_coefs_Y[][64], float blockified_img_Y[][8][8], int N_block){
    float tmp_Y[8];
    float res_Y[8][8];
    int idx, N, i, j;
    for (N=0; N<N_block; N++){
        for (i=0; i<8; i++){
            for (j=0; j<8; j++){
                idx = ZIGZAG[i][j];
                blockified_img_Y[N][i][j] = seq_dct_coefs_Y[N][idx];
            }
        }
    }
}
