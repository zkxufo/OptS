#include <map>
#include <ctime>
#include <chrono>
#include "../Utils/utils.h"
#include "../Utils/Q_Table.h"
#include "../EntCoding/Huffman.h"
using namespace std;
const double MIN_Q_VAL = 1;
class HDQ{
    public:
        // attributes
        double Beta_S;
        double Beta_W;
        double Beta_X;
        double Q_table_Y[64];
        double Q_table_C[64];
        int seq_len_Y, seq_len_C; // # 8x8 DCT blocks after subsampling
        int n_row;
        int n_col;
        // BLOCK Block;
        int seq_block_dcts[64];
        int DCT_block_shape[3];
        int img_shape_Y[2], img_shape_C[2]; // size of channels after subsampling
        //TODO: P_DC for DC coefficient
        map<int, double> P_DC_Y;
        map<int, double> P_DC_C;
        double J_Y = 10e10;
        double J_C = 10e10;
        int QF_C;
        int QF_Y;
        int J, a, b;
        double Loss;
        double EntACY = 0;
        double EntACC = 0;
        double EntDCY = 0;
        double EntDCC = 0;
        vector<int> RSlst;
        vector<int> IDlst;
        void __init__(double eps, double Beta_S, double Beta_W, double Beta_X,
                      double Lmbda, int QF_Y, int QF_C, 
                      int J, int a, int b);
        double __call__(vector<vector<vector<double>>>& image);
};

void HDQ::__init__(double eps, double Beta_S, double Beta_W, double Beta_X,
                   double Lmbda, int QF_Y, int QF_C, 
                   int J, int a, int b){
    HDQ::RSlst.reserve(64);
    HDQ::IDlst.reserve(64);
    HDQ::RSlst.clear();
    HDQ::IDlst.clear();
    HDQ::Beta_S = Beta_S;
    HDQ::Beta_W = Beta_W;
    HDQ::Beta_X = Beta_X;
    quantizationTable(QF_Y, true, HDQ::Q_table_Y);
    quantizationTable(QF_C, false, HDQ::Q_table_C);
    HDQ::J_Y = 10e10;
    HDQ::J_C = 10e10;
    HDQ::J = J;
    HDQ::a = a;
    HDQ::b = b;
}

double HDQ::__call__(vector<vector<vector<double>>>& image){
    int i,j,k,l,r,s;
    int BITS[33];
    int size;
    std::fill_n(BITS, 33, 0);
    shape(image, HDQ::img_shape_Y);
    HDQ::n_col = HDQ::img_shape_Y[1];
    HDQ::n_row = HDQ::img_shape_Y[0];
    HDQ::seq_len_Y = pad_shape(HDQ::img_shape_Y[0], 8)*pad_shape(HDQ::img_shape_Y[1], 8)/64;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int pad_rows = pad_shape(HDQ::img_shape_Y[0], J);
    int pad_cols = pad_shape(HDQ::img_shape_Y[1], J);
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    Subsampling(image[1], HDQ::img_shape_Y, HDQ::img_shape_C, HDQ::J, HDQ::a, HDQ::b);
    Subsampling(image[2], HDQ::img_shape_Y, HDQ::img_shape_C, HDQ::J, HDQ::a, HDQ::b);

    HDQ::seq_len_C = pad_shape(Smplcols, 8)*pad_shape(Smplrows, 8)/64;
    auto blockified_img_Y = new double[HDQ::seq_len_Y][8][8];
    auto blockified_img_Cb = new double[HDQ::seq_len_C][8][8];
    auto blockified_img_Cr = new double[HDQ::seq_len_C][8][8];

    auto seq_dct_coefs_Y = new double[HDQ::seq_len_Y][64];
    auto seq_dct_coefs_Cb = new double[HDQ::seq_len_C][64];
    auto seq_dct_coefs_Cr = new double[HDQ::seq_len_C][64];

    auto seq_dct_idxs_Y = new double[HDQ::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new double[HDQ::seq_len_C][64];
    auto seq_dct_idxs_Cr = new double[HDQ::seq_len_C][64];

    auto DC_idxs_Y = new double[HDQ::seq_len_Y];
    auto DC_idxs_Cb = new double[HDQ::seq_len_C];
    auto DC_idxs_Cr = new double[HDQ::seq_len_C];

    blockify(image[0], HDQ::img_shape_Y, blockified_img_Y);
    blockify(image[1], HDQ::img_shape_C, blockified_img_Cb);    
    blockify(image[2], HDQ::img_shape_C, blockified_img_Cr);

    block_2_seqdct(blockified_img_Y, seq_dct_coefs_Y, HDQ::seq_len_Y);
    block_2_seqdct(blockified_img_Cb, seq_dct_coefs_Cb, HDQ::seq_len_C);
    block_2_seqdct(blockified_img_Cr, seq_dct_coefs_Cr, HDQ::seq_len_C);

    Quantize(seq_dct_coefs_Y,seq_dct_idxs_Y, 
             HDQ::Q_table_Y, HDQ::seq_len_Y);
    Quantize(seq_dct_coefs_Cb,seq_dct_idxs_Cb,
             HDQ::Q_table_C,HDQ::seq_len_C);
    Quantize(seq_dct_coefs_Cr,seq_dct_idxs_Cr,
             HDQ::Q_table_C,HDQ::seq_len_C);
    
    map<int, double> DC_P;
    map<int, double> AC_Y_P;
    map<int, double> AC_C_P;
    DC_P.clear();
    double EntDCY=0;
    double EntDCC=0;
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, HDQ::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, HDQ::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    EntDCY = calHuffmanCodeSize(DC_P);
    DC_P.clear();
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, HDQ::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, HDQ::seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, HDQ::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, HDQ::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    DC_P.clear();

    for(i=0; i<HDQ::seq_len_Y; i++){
        HDQ::RSlst.clear(); HDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], HDQ::RSlst, HDQ::IDlst);
        cal_P_from_RSlst(HDQ::RSlst, AC_Y_P);
    }
    HDQ::RSlst.clear(); HDQ::IDlst.clear();
    AC_Y_P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(AC_Y_P);
    AC_Y_P.clear();

    for(i=0; i<HDQ::seq_len_C; i++){
        HDQ::RSlst.clear(); HDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], HDQ::RSlst, HDQ::IDlst);
        cal_P_from_RSlst(HDQ::RSlst, AC_C_P);
        HDQ::RSlst.clear(); HDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], HDQ::RSlst, HDQ::IDlst);
        cal_P_from_RSlst(HDQ::RSlst, AC_C_P);
    }
    AC_C_P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(AC_C_P);
    double BPP=0;
    double file_size = EntACC+EntACY+EntDCC+EntDCY; // Run_length coding
    file_size += 8*(1+1); // SOI
    file_size += 8*(1+1+2+5+1+1+2+2+1); // APP0
    file_size += 8*(1+1+2+1+1+64); // DQT
    file_size += 8*(1+1+2+1+2+2+1+1+1+1); // SOF0
    // TODO: cal n in DHT and change 2 to 4
    file_size += 8*(1+1+2+1+16)*2+256*2; //DHT
    file_size += 8*(1+1+2+1+1+1+3); // SOS
    file_size += 8*(2+2+1+2);
    file_size += 8*(1+1); //EOI
    BPP = file_size/HDQ::img_shape_Y[0]/HDQ::img_shape_Y[1];
    delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    Dequantize(seq_dct_idxs_Y, HDQ::Q_table_Y, HDQ::seq_len_Y); //seq_dct_idxs_Y: [][64]
    Dequantize(seq_dct_idxs_Cb, HDQ::Q_table_C, HDQ::seq_len_C);
    Dequantize(seq_dct_idxs_Cr, HDQ::Q_table_C, HDQ::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Y, blockified_img_Y, HDQ::seq_len_Y); //seq_dct_idxs_Y: [][8[8]
    seq_2_blockidct(seq_dct_idxs_Cb, blockified_img_Cb, HDQ::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Cr, blockified_img_Cr, HDQ::seq_len_C);
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    deblockify(blockified_img_Y,  image[0], HDQ::img_shape_Y); //seq_dct_idxs_Y: [][], crop here!
    deblockify(blockified_img_Cb, image[1], HDQ::img_shape_C);
    deblockify(blockified_img_Cr, image[2], HDQ::img_shape_C);
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    Upsampling(image[1], HDQ::img_shape_Y, HDQ::J, HDQ::a, HDQ::b);
    Upsampling(image[2], HDQ::img_shape_Y, HDQ::J, HDQ::a, HDQ::b);
    return BPP;
}