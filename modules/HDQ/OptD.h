#include <map>
#include <ctime>
#include <chrono>
#include "../Utils/utils.h"
#include "../Utils/Q_Table.h"
#include "../EntCoding/Huffman.h"
#include <limits>
// #include <algorithm>
// #include <math.h>

using namespace std;
const float MIN_Q_VAL = 1;
class OptD{
    public:
        // attributes
        float Q_table_Y[64];
        float Q_table_C[64];
        int seq_len_Y, seq_len_C; // # 8x8 DCT blocks after subsampling
        int n_row;
        int n_col;
        // BLOCK Block;
        int seq_block_dcts[64];
        int DCT_block_shape[3];
        int img_shape_Y[2], img_shape_C[2]; // size of channels after subsampling
        //TODO: P_DC for DC coefficient
        map<int, float> P_DC_Y;
        map<int, float> P_DC_C;
        float J_Y = 10e10;
        float J_C = 10e10;
        int QF_C;
        int QF_Y;
        int J, a, b;
        int QMAX_Y, QMAX_C;
        float DT_Y, DT_C;
        float d_waterlevel_Y, d_waterlevel_C;
        float Loss;
        float EntACY = 0;
        float EntACC = 0;
        float EntDCY = 0;
        float EntDCC = 0;
        float (*Sen_Map)[64];
        float MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE;
        vector<int> RSlst;
        vector<int> IDlst;
        void __init__(float Sen_Map[3][64], int QF_Y, int QF_C, 
                      int J, int a, int b, float DT_Y, float DT_C, 
                      float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C);
        float __call__(vector<vector<vector<float>>>& image, vector<float>& q_table);
};

void OptD::__init__(float Sen_Map[3][64], int QF_Y, int QF_C, 
                   int J, int a, int b, float DT_Y, float DT_C,
                   float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C){
    OptD::RSlst.reserve(64);
    OptD::IDlst.reserve(64);
    OptD::RSlst.clear();
    OptD::IDlst.clear();
    quantizationTable(QF_Y, true, OptD::Q_table_Y);
    quantizationTable(QF_C, false, OptD::Q_table_C);
    OptD::J_Y = 10e10;
    OptD::J_C = 10e10;
    OptD::J = J;
    OptD::a = a;
    OptD::b = b;
    OptD::DT_Y = DT_Y;
    OptD::DT_C = DT_C;
    OptD::d_waterlevel_Y = d_waterlevel_Y;
    OptD::QMAX_Y = QMAX_Y;
    OptD::d_waterlevel_C = d_waterlevel_C;
    OptD::QMAX_C = QMAX_C;
    OptD::Sen_Map = Sen_Map;
}

float OptD::__call__(vector<vector<vector<float>>>& image, vector<float>& q_table){
    int i,j,k,l,r,s;
    int BITS[33];
    int size;
    std::fill_n(BITS, 33, 0);
    shape(image, OptD::img_shape_Y);
    OptD::n_col = OptD::img_shape_Y[1];
    OptD::n_row = OptD::img_shape_Y[0];
    OptD::seq_len_Y = pad_shape(OptD::img_shape_Y[0], 8)*pad_shape(OptD::img_shape_Y[1], 8)/64;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int pad_rows = pad_shape(OptD::img_shape_Y[0], J);
    int pad_cols = pad_shape(OptD::img_shape_Y[1], J);
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    Subsampling(image[1], OptD::img_shape_Y, OptD::img_shape_C, OptD::J, OptD::a, OptD::b);
    Subsampling(image[2], OptD::img_shape_Y, OptD::img_shape_C, OptD::J, OptD::a, OptD::b);

    OptD::seq_len_C = pad_shape(Smplcols, 8)*pad_shape(Smplrows, 8)/64;
    auto blockified_img_Y = new float[OptD::seq_len_Y][8][8];
    auto blockified_img_Cb = new float[OptD::seq_len_C][8][8];
    auto blockified_img_Cr = new float[OptD::seq_len_C][8][8];

    auto seq_dct_coefs_Y = new float[OptD::seq_len_Y][64];
    auto seq_dct_coefs_Cb = new float[OptD::seq_len_C][64];
    auto seq_dct_coefs_Cr = new float[OptD::seq_len_C][64];

    auto seq_dct_idxs_Y = new float[OptD::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new float[OptD::seq_len_C][64];
    auto seq_dct_idxs_Cr = new float[OptD::seq_len_C][64];

    auto DC_idxs_Y = new float[OptD::seq_len_Y];
    auto DC_idxs_Cb = new float[OptD::seq_len_C];
    auto DC_idxs_Cr = new float[OptD::seq_len_C];

    blockify(image[0], OptD::img_shape_Y, blockified_img_Y);
    blockify(image[1], OptD::img_shape_C, blockified_img_Cb);    
    blockify(image[2], OptD::img_shape_C, blockified_img_Cr);

    block_2_seqdct(blockified_img_Y, seq_dct_coefs_Y, OptD::seq_len_Y);
    block_2_seqdct(blockified_img_Cb, seq_dct_coefs_Cb, OptD::seq_len_C);
    block_2_seqdct(blockified_img_Cr, seq_dct_coefs_Cr, OptD::seq_len_C);


    // // adaptive Qmax 
    // float max_q = std::numeric_limits<float>::min(), tmp, num1, num2;
    // for(int N=0; N<OptD::seq_len_C; N++)
    // {
    //     tmp = *max_element(seq_dct_coefs_Y[N], seq_dct_coefs_Y[N] + 64);
    //     if (tmp > max_q) max_q = tmp;
    // }
    // OptD::QMAX_Y = 2 * max_q + 1;
    // cout << "Max Quantization Step Y : " <<OptD::QMAX_Y << endl;

    // max_q = std::numeric_limits<float>::min();
    // for(int N=0; N<OptD::seq_len_C; N++)
    // {
    //     num1 = *max_element(seq_dct_coefs_Cb[N], seq_dct_coefs_Cb[N] + 64);
    //     num2 = *max_element(seq_dct_coefs_Cr[N], seq_dct_coefs_Cr[N] + 64);
    //     tmp = max(num1, num2);
    //     if (tmp > max_q) max_q = tmp;
    // }
    // OptD::QMAX_C = 2 * max_q + 1;
    // cout << "Max Quantization Step CbCr : " <<OptD::QMAX_C << endl;

    float dummy;
    // Customized Quantization Table
    quantizationTable_OptD_Y(OptD::Sen_Map, seq_dct_coefs_Y, OptD::Q_table_Y, 
                OptD::seq_len_Y, OptD::DT_Y, OptD::d_waterlevel_Y, OptD::QMAX_Y, dummy);
    // cout << "DT_Y = " << OptD::DT_Y << "\t" << "d_waterLevel_Y = " << OptD::d_waterlevel_Y << endl;
    quantizationTable_OptD_C(OptD::Sen_Map, seq_dct_coefs_Cb, seq_dct_coefs_Cr, OptD::Q_table_C
        , OptD::seq_len_C, OptD::DT_C, OptD::d_waterlevel_C, OptD::QMAX_C, dummy);
    // cout << "DT_C = " << OptD::DT_C << "\t" << "d_waterLevel_C = " << OptD::d_waterlevel_C << endl;
   
    int check = checkQmax(OptD::Q_table_Y, OptD::QMAX_Y, OptD::Q_table_C, OptD::QMAX_C);
    if (check == 2)
    {
        check = -1;
    }
    else
    {
        check = 1;
    }
    //

    for(int j=0; j<64; ++j)
    {
    q_table[j] = OptD::Q_table_Y[j];
    }

    for(int j=0; j<64; ++j){
    q_table[64+j] = OptD::Q_table_C[j];
    }

    Quantize(seq_dct_coefs_Y,seq_dct_idxs_Y, 
             OptD::Q_table_Y, OptD::seq_len_Y);
    Quantize(seq_dct_coefs_Cb,seq_dct_idxs_Cb,
             OptD::Q_table_C,OptD::seq_len_C);
    Quantize(seq_dct_coefs_Cr,seq_dct_idxs_Cr,
             OptD::Q_table_C,OptD::seq_len_C);

    
    // cout << "number of blocks: " << seq_len_C << endl;
    // cout << "Cb indices" << endl;
    // for(int N=0; N<OptD::seq_len_C; N++)
    // {
    //     cout << seq_dct_idxs_Cb[N][33] << endl;
    // }
    // cout << "Cr indices" << endl;
    // for(int N=0; N<OptD::seq_len_C; N++)
    // {
    //     cout << seq_dct_idxs_Cr[N][33] << endl;
    // }
    
    map<int, float> DC_P;
    map<int, float> AC_Y_P;
    map<int, float> AC_C_P;
    DC_P.clear();
    float EntDCY=0;
    float EntDCC=0;
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, OptD::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, OptD::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    EntDCY = calHuffmanCodeSize(DC_P);
    DC_P.clear();
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, OptD::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, OptD::seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, OptD::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, OptD::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    DC_P.clear();

    for(i=0; i<OptD::seq_len_Y; i++){
        OptD::RSlst.clear(); OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], OptD::RSlst, OptD::IDlst);
        cal_P_from_RSlst(OptD::RSlst, AC_Y_P);
    }
    OptD::RSlst.clear(); OptD::IDlst.clear();
    AC_Y_P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(AC_Y_P);
    AC_Y_P.clear();

    for(i=0; i<OptD::seq_len_C; i++){
        OptD::RSlst.clear(); OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], OptD::RSlst, OptD::IDlst);
        cal_P_from_RSlst(OptD::RSlst, AC_C_P);
        OptD::RSlst.clear(); OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], OptD::RSlst, OptD::IDlst);
        cal_P_from_RSlst(OptD::RSlst, AC_C_P);
    }
    AC_C_P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(AC_C_P);
    float BPP=0;
    float file_size = EntACC+EntACY+EntDCC+EntDCY+FLAG_SIZE; // Run_length coding
    BPP = file_size/OptD::img_shape_Y[0]/OptD::img_shape_Y[1];
    delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    Dequantize(seq_dct_idxs_Y, OptD::Q_table_Y, OptD::seq_len_Y); //seq_dct_idxs_Y: [][64]
    Dequantize(seq_dct_idxs_Cb, OptD::Q_table_C, OptD::seq_len_C);
    Dequantize(seq_dct_idxs_Cr, OptD::Q_table_C, OptD::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Y, blockified_img_Y, OptD::seq_len_Y); //seq_dct_idxs_Y: [][8[8]
    seq_2_blockidct(seq_dct_idxs_Cb, blockified_img_Cb, OptD::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Cr, blockified_img_Cr, OptD::seq_len_C);
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    deblockify(blockified_img_Y,  image[0], OptD::img_shape_Y); //seq_dct_idxs_Y: [][], crop here!
    deblockify(blockified_img_Cb, image[1], OptD::img_shape_C);
    deblockify(blockified_img_Cr, image[2], OptD::img_shape_C);
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    Upsampling(image[1], OptD::img_shape_Y, OptD::J, OptD::a, OptD::b);
    Upsampling(image[2], OptD::img_shape_Y, OptD::J, OptD::a, OptD::b);
    return (check * BPP);
}
