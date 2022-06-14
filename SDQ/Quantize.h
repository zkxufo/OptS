#include <math.h>
#include <iostream>
using namespace std;
void Dequantize(double seq_dct_idxs[][64], double Q_table[64], int N_block){
    cout<<"Dequantize"<<N_block<<endl;
    for(int N=0; N<N_block; N++){
        for(int x=0; x<64; x++){
            seq_dct_idxs[N][x] = seq_dct_idxs[N][x]*Q_table[x];
            // seq_dct_idxs[N][x] = seq_dct_idxs[N][x]*20;
        }
    }
}

void Quantize(double seq_dct_coefs[][64], double seq_dct_idxs[][64],
              double Q_table[64],  int N_block){
    cout<<"Quantize"<<N_block<<endl;
    int N, x;
    for (N=0; N<N_block; N++){
        for(int x=0; x<64; x++){
            seq_dct_idxs[N][x] = round(seq_dct_coefs[N][x]/Q_table[x]);
            // seq_dct_idxs[N][x] = round(seq_dct_coefs[N][x]/20);
        }
    }
}
