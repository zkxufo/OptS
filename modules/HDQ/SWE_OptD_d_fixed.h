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
class HDQ_OptD{
    public:
        // attributes
        float Q_table_Y[64];
        float Q_table_C[64];
        float varianceData_Y[64];
        float varianceData_CbCr[128];
        float lambdaData_Cb[64],  lambdaData_Cr[64];  
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
        float max_var_Y = 0;
        float max_var_C = 0;
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
        float eps = 1e-4, iter_stop = 50;
        float up, low, mid;

        float (*Sen_Map)[64];
        // float MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE;
        vector<int> RSlst;
        vector<int> IDlst;
        void __init__(float Sen_Map[3][64], int QF_Y, int QF_C, 
                      int J, int a, int b, float DT_Y, float DT_C, 
                      float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C);
        float __call__(vector<vector<vector<float>>>& image, vector<float>& q_table);
        float SWE_eval(int Sens_index, float Q_table[64],float seq_dct_coefs[][64], float seq_dct_idxs[][64], int seq_len, float d_waterlevel_C);

};

void HDQ_OptD::__init__(float Sen_Map[3][64], int QF_Y, int QF_C, 
                   int J, int a, int b, float DT_Y, float DT_C,
                   float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C){
    HDQ_OptD::RSlst.reserve(64);
    HDQ_OptD::IDlst.reserve(64);
    HDQ_OptD::RSlst.clear();
    HDQ_OptD::IDlst.clear();
    quantizationTable(QF_Y, true, HDQ_OptD::Q_table_Y);
    quantizationTable(QF_C, false, HDQ_OptD::Q_table_C);
    HDQ_OptD::J_Y = 10e10;
    HDQ_OptD::J_C = 10e10;
    HDQ_OptD::J = J;
    HDQ_OptD::a = a;
    HDQ_OptD::b = b;
    HDQ_OptD::QF_Y = QF_Y;
    HDQ_OptD::QF_C = QF_C;
    HDQ_OptD::DT_Y = DT_Y;
    HDQ_OptD::DT_C = DT_C;
    HDQ_OptD::d_waterlevel_Y = d_waterlevel_Y;
    HDQ_OptD::QMAX_Y = QMAX_Y;
    HDQ_OptD::d_waterlevel_C = d_waterlevel_C;
    HDQ_OptD::QMAX_C = QMAX_C;
    HDQ_OptD::Sen_Map = Sen_Map;
}

float HDQ_OptD::__call__(vector<vector<vector<float>>>& image, vector<float>& q_table){
    int i,j,k,l,r,s;
    int BITS[33];
    int size;
    std::fill_n(BITS, 33, 0);
    shape(image, HDQ_OptD::img_shape_Y);
    HDQ_OptD::n_col = HDQ_OptD::img_shape_Y[1];
    HDQ_OptD::n_row = HDQ_OptD::img_shape_Y[0];
    HDQ_OptD::seq_len_Y = pad_shape(HDQ_OptD::img_shape_Y[0], 8)*pad_shape(HDQ_OptD::img_shape_Y[1], 8)/64;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int pad_rows = pad_shape(HDQ_OptD::img_shape_Y[0], J);
    int pad_cols = pad_shape(HDQ_OptD::img_shape_Y[1], J);
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    Subsampling(image[1], HDQ_OptD::img_shape_Y, HDQ_OptD::img_shape_C, HDQ_OptD::J, HDQ_OptD::a, HDQ_OptD::b);
    Subsampling(image[2], HDQ_OptD::img_shape_Y, HDQ_OptD::img_shape_C, HDQ_OptD::J, HDQ_OptD::a, HDQ_OptD::b);

    HDQ_OptD::seq_len_C = pad_shape(Smplcols, 8)*pad_shape(Smplrows, 8)/64;
    auto blockified_img_Y = new float[HDQ_OptD::seq_len_Y][8][8];
    auto blockified_img_Cb = new float[HDQ_OptD::seq_len_C][8][8];
    auto blockified_img_Cr = new float[HDQ_OptD::seq_len_C][8][8];

    auto seq_dct_coefs_Y = new float[HDQ_OptD::seq_len_Y][64];
    auto seq_dct_coefs_Cb = new float[HDQ_OptD::seq_len_C][64];
    auto seq_dct_coefs_Cr = new float[HDQ_OptD::seq_len_C][64];


    auto seq_dct_idxs_Y = new float[HDQ_OptD::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new float[HDQ_OptD::seq_len_C][64];
    auto seq_dct_idxs_Cr = new float[HDQ_OptD::seq_len_C][64];

    auto DC_idxs_Y = new float[HDQ_OptD::seq_len_Y];
    auto DC_idxs_Cb = new float[HDQ_OptD::seq_len_C];
    auto DC_idxs_Cr = new float[HDQ_OptD::seq_len_C];

    blockify(image[0], HDQ_OptD::img_shape_Y, blockified_img_Y);
    blockify(image[1], HDQ_OptD::img_shape_C, blockified_img_Cb);    
    blockify(image[2], HDQ_OptD::img_shape_C, blockified_img_Cr);


    // ORG --> seq
    block_2_seqdct(blockified_img_Y, seq_dct_coefs_Y, HDQ_OptD::seq_len_Y);
    block_2_seqdct(blockified_img_Cb, seq_dct_coefs_Cb, HDQ_OptD::seq_len_C);
    block_2_seqdct(blockified_img_Cr, seq_dct_coefs_Cr, HDQ_OptD::seq_len_C);


    // // SWE
    // cout << "A: " << DT_Y << endl;

    quantizationTable_OptD_Y(HDQ_OptD::Sen_Map, seq_dct_coefs_Y, HDQ_OptD::Q_table_Y, HDQ_OptD::varianceData_Y,
                HDQ_OptD::seq_len_Y, HDQ_OptD::DT_Y, HDQ_OptD::d_waterlevel_Y, HDQ_OptD::QMAX_Y, HDQ_OptD::max_var_Y);

    quantizationTable_OptD_C(HDQ_OptD::Sen_Map, seq_dct_coefs_Cb, seq_dct_coefs_Cr, HDQ_OptD::Q_table_C, HDQ_OptD::varianceData_CbCr
            , HDQ_OptD::seq_len_C, HDQ_OptD::DT_C, HDQ_OptD::d_waterlevel_C, HDQ_OptD::QMAX_C, HDQ_OptD::max_var_C);
    
    float SWE_Y_target = SWE_eval(0,HDQ_OptD::Q_table_Y, seq_dct_coefs_Y, seq_dct_idxs_Y, HDQ_OptD::seq_len_Y, HDQ_OptD::d_waterlevel_Y);
    float SWE_C_target = SWE_eval(1,HDQ_OptD::Q_table_C, seq_dct_coefs_Cb, seq_dct_idxs_Cb, HDQ_OptD::seq_len_C, HDQ_OptD::d_waterlevel_C);
    SWE_C_target += SWE_eval(2,HDQ_OptD::Q_table_C, seq_dct_coefs_Cr, seq_dct_idxs_Cr, HDQ_OptD::seq_len_C, HDQ_OptD::d_waterlevel_C);

    float Sen_Map_ones[3][64];
    Sens_ones(Sen_Map_ones);                // all ones senmap for quantizationTable_OptD
    // pass the senstivity full of ones to this function to force return back the original value 
    // of the variance without scaling it with the senstivity ==> OptD only
    cal_ImageStat_CbCr(Sen_Map_ones, seq_dct_coefs_Cb, seq_dct_coefs_Cr, HDQ_OptD::varianceData_CbCr,
                    HDQ_OptD::lambdaData_Cb, HDQ_OptD::lambdaData_Cr, HDQ_OptD::seq_len_C, HDQ_OptD::max_var_C);
    
    // cout << "A : " << SWE_Y << " -- " << SWE_C << endl;

    // Customized Quantization Table
    // ## [Important] ##
    // SWE_Y_target from JPEG is clipped by [0, sum of variance] from the prespective of OptD
    // to guarantee that this number is reachable by OptD 


    quantizationTable_OptD_Y(Sen_Map_ones, seq_dct_coefs_Y, HDQ_OptD::Q_table_Y, HDQ_OptD::varianceData_Y,
                HDQ_OptD::seq_len_Y, SWE_Y_target, HDQ_OptD::d_waterlevel_Y, HDQ_OptD::QMAX_Y, HDQ_OptD::max_var_Y);

    HDQ_OptD::up = HDQ_OptD::max_var_Y;
    HDQ_OptD::low = 0; //HDQ_OptD::d_waterlevel_Y;
    float mid_D, low_D;

    // cout << "Target SWE_Y = " << SWE_Y_target << endl;

    // cout << "up_d SWE_Y = " << HDQ_OptD::up  << endl;
    // cout << "low_d SWE_Y = " << HDQ_OptD::low   << endl;

    int count = 0;

    while (((HDQ_OptD::up-HDQ_OptD::low) >= HDQ_OptD::eps) && (count < iter_stop))
    {
        HDQ_OptD::mid = (HDQ_OptD::up + HDQ_OptD::low)/2.0;
        // mid
        quantizationTable_OptD_Y(Sen_Map_ones, seq_dct_coefs_Y, HDQ_OptD::Q_table_Y, HDQ_OptD::varianceData_Y,
                    HDQ_OptD::seq_len_Y, HDQ_OptD::DT_Y, HDQ_OptD::mid, HDQ_OptD::QMAX_Y, HDQ_OptD::max_var_Y);
        mid_D = SWE_eval(0,HDQ_OptD::Q_table_Y, seq_dct_coefs_Y, seq_dct_idxs_Y, HDQ_OptD::seq_len_Y, HDQ_OptD::d_waterlevel_Y);


        // low
        quantizationTable_OptD_Y(Sen_Map_ones, seq_dct_coefs_Y, HDQ_OptD::Q_table_Y, HDQ_OptD::varianceData_Y,
                    HDQ_OptD::seq_len_Y, HDQ_OptD::DT_Y, HDQ_OptD::low, HDQ_OptD::QMAX_Y, HDQ_OptD::max_var_Y);
        low_D = SWE_eval(0,HDQ_OptD::Q_table_Y, seq_dct_coefs_Y, seq_dct_idxs_Y, HDQ_OptD::seq_len_Y, HDQ_OptD::d_waterlevel_Y);
        
        // cout << "iter d_Y = " << HDQ_OptD::mid << "\t mid_D=" << mid_D << "\t low_D=" << low_D  << endl;
        if (mid_D == SWE_Y_target)
        {
            break;
        }
        else if ( (mid_D - SWE_Y_target) * (low_D - SWE_Y_target) <= 0 )
        {
            HDQ_OptD::up = HDQ_OptD::mid;
        }
        else
        {
            HDQ_OptD::low = HDQ_OptD::mid;
        }
        count++;

        // cout << "DT_Y = " << HDQ_OptD::DT_Y << "\t" << "d_waterLevel_Y = " << HDQ_OptD::d_waterlevel_Y << endl;

    }
    quantizationTable_OptD_Y(Sen_Map_ones, seq_dct_coefs_Y, HDQ_OptD::Q_table_Y, HDQ_OptD::varianceData_Y,
            HDQ_OptD::seq_len_Y, HDQ_OptD::DT_Y, HDQ_OptD::mid, HDQ_OptD::QMAX_Y, HDQ_OptD::max_var_Y);
    mid_D = SWE_eval(0,HDQ_OptD::Q_table_Y, seq_dct_coefs_Y, seq_dct_idxs_Y, HDQ_OptD::seq_len_Y, HDQ_OptD::d_waterlevel_Y);

    // cout << "Selected d_Y = " << HDQ_OptD::mid  << endl;
    // cout << "Selected SWE_Y = " << mid_D  << endl;




// ------------------------------------- CbCr -----------------------------------------------

    // quantizationTable_OptD_C(Sen_Map_ones, seq_dct_coefs_Cb, seq_dct_coefs_Cr, HDQ_OptD::Q_table_C, HDQ_OptD::varianceData_CbCr
    //     , HDQ_OptD::seq_len_C, SWE_C_target, HDQ_OptD::d_waterlevel_C, HDQ_OptD::QMAX_C, HDQ_OptD::max_var_C);
    
    OptD_C(Sen_Map_ones, HDQ_OptD::varianceData_CbCr, HDQ_OptD::lambdaData_Cb, 
            HDQ_OptD::lambdaData_Cr, HDQ_OptD::Q_table_C, SWE_C_target, HDQ_OptD::d_waterlevel_C, HDQ_OptD::QMAX_C);

    HDQ_OptD::up = HDQ_OptD::max_var_C;
    HDQ_OptD::low = 0; //HDQ_OptD::d_waterlevel_C;

    // cout << "Target SWE_C = " << SWE_C_target << endl;

    // cout << "up_d SWE_C = " << HDQ_OptD::up  << endl;
    // cout << "low_d SWE_C = " << HDQ_OptD::low   << endl;
    
    count = 0;
    while (((HDQ_OptD::up-HDQ_OptD::low) >= HDQ_OptD::eps) && (count < iter_stop))
    {
        HDQ_OptD::mid = (HDQ_OptD::up + HDQ_OptD::low)/2.0;

        // quantizationTable_OptD_C(Sen_Map_ones, seq_dct_coefs_Cb, seq_dct_coefs_Cr, HDQ_OptD::Q_table_C, HDQ_OptD::varianceData_CbCr
        //     , HDQ_OptD::seq_len_C, HDQ_OptD::DT_C, HDQ_OptD::mid, HDQ_OptD::QMAX_C, HDQ_OptD::max_var_C);
        
        HDQ_OptD::d_waterlevel_C = HDQ_OptD::mid;
        OptD_C(Sen_Map_ones, HDQ_OptD::varianceData_CbCr, HDQ_OptD::lambdaData_Cb, 
                HDQ_OptD::lambdaData_Cr, HDQ_OptD::Q_table_C,  HDQ_OptD::DT_C, HDQ_OptD::mid, HDQ_OptD::QMAX_C);

        
        mid_D = SWE_eval(1,HDQ_OptD::Q_table_C, seq_dct_coefs_Cb, seq_dct_idxs_Cb, HDQ_OptD::seq_len_C, HDQ_OptD::d_waterlevel_C);
        mid_D += SWE_eval(2,HDQ_OptD::Q_table_C, seq_dct_coefs_Cr, seq_dct_idxs_Cr, HDQ_OptD::seq_len_C, HDQ_OptD::d_waterlevel_C);

        // print_Q_table(HDQ_OptD::Q_table_C);

        // quantizationTable_OptD_C(Sen_Map_ones, seq_dct_coefs_Cb, seq_dct_coefs_Cr, HDQ_OptD::Q_table_C, HDQ_OptD::varianceData_CbCr
        //     , HDQ_OptD::seq_len_C, HDQ_OptD::DT_C, HDQ_OptD::low, HDQ_OptD::QMAX_C, HDQ_OptD::max_var_C);
        
        HDQ_OptD::d_waterlevel_C = HDQ_OptD::low;
        OptD_C(Sen_Map_ones, HDQ_OptD::varianceData_CbCr, HDQ_OptD::lambdaData_Cb, 
                HDQ_OptD::lambdaData_Cr, HDQ_OptD::Q_table_C, HDQ_OptD::DT_C, HDQ_OptD::low, HDQ_OptD::QMAX_C);

        low_D = SWE_eval(1,HDQ_OptD::Q_table_C, seq_dct_coefs_Cb, seq_dct_idxs_Cb, HDQ_OptD::seq_len_C, HDQ_OptD::d_waterlevel_C);
        low_D += SWE_eval(2,HDQ_OptD::Q_table_C, seq_dct_coefs_Cr, seq_dct_idxs_Cr, HDQ_OptD::seq_len_C, HDQ_OptD::d_waterlevel_C);

        // print_Q_table(HDQ_OptD::Q_table_C);

        // cout << "iter d_C = " << HDQ_OptD::mid << "\t mid_D=" << mid_D << "\t low_D=" << low_D  << endl;
        if (mid_D == SWE_C_target)
        {
            break;
        }
        else if ( (mid_D - SWE_C_target) * (low_D - SWE_C_target) <= 0 )
        {
            HDQ_OptD::up = HDQ_OptD::mid;
        }
        else
        {
            HDQ_OptD::low = HDQ_OptD::mid;
        }
        count++;
        // cout << "DT_Y = " << HDQ_OptD::DT_Y << "\t" << "d_waterLevel_Y = " << HDQ_OptD::d_waterlevel_Y << endl;

    }
    // quantizationTable_OptD_C(Sen_Map_ones, seq_dct_coefs_Cb, seq_dct_coefs_Cr, HDQ_OptD::Q_table_C, HDQ_OptD::varianceData_CbCr
    //     , HDQ_OptD::seq_len_C, HDQ_OptD::DT_C, HDQ_OptD::mid , HDQ_OptD::QMAX_C, HDQ_OptD::max_var_C);

    HDQ_OptD::d_waterlevel_C = HDQ_OptD::mid;
    OptD_C(Sen_Map_ones, HDQ_OptD::varianceData_CbCr, HDQ_OptD::lambdaData_Cb, 
            HDQ_OptD::lambdaData_Cr, HDQ_OptD::Q_table_C, HDQ_OptD::DT_C, HDQ_OptD::mid, HDQ_OptD::QMAX_C);
    
    
    mid_D = SWE_eval(1,HDQ_OptD::Q_table_C, seq_dct_coefs_Cb, seq_dct_idxs_Cb, HDQ_OptD::seq_len_C, HDQ_OptD::d_waterlevel_C);
    mid_D += SWE_eval(2,HDQ_OptD::Q_table_C, seq_dct_coefs_Cr, seq_dct_idxs_Cr, HDQ_OptD::seq_len_C, HDQ_OptD::d_waterlevel_C);
 
    // cout << "Selected d_C = " << HDQ_OptD::mid  << endl;
    // cout << "Selected SWE_C = " << mid_D  << endl;



    // Fast Quantization check
    int check = checkQmax(HDQ_OptD::Q_table_Y, HDQ_OptD::QMAX_Y, HDQ_OptD::Q_table_C, HDQ_OptD::QMAX_C);
    if (check == 2)
    {
        check = -1;
    }
    else
    {
        check = 1;
    }

    Quantize(seq_dct_coefs_Y,seq_dct_idxs_Y, 
             HDQ_OptD::Q_table_Y, HDQ_OptD::seq_len_Y);
    Quantize(seq_dct_coefs_Cb,seq_dct_idxs_Cb,
             HDQ_OptD::Q_table_C,HDQ_OptD::seq_len_C);
    Quantize(seq_dct_coefs_Cr,seq_dct_idxs_Cr,
             HDQ_OptD::Q_table_C,HDQ_OptD::seq_len_C);

    fast_quatization_CbCr(3 , HDQ_OptD::varianceData_CbCr, seq_dct_idxs_Cb, 
                            seq_dct_idxs_Cr, HDQ_OptD::d_waterlevel_C, HDQ_OptD::seq_len_C);


    for(int j=0; j<64; ++j)
    {
    q_table[j] = HDQ_OptD::Q_table_Y[j];
    }

    for(int j=0; j<64; ++j){
    q_table[64+j] = HDQ_OptD::Q_table_C[j];
    }
    
    map<int, float> DC_P;
    map<int, float> AC_Y_P;
    map<int, float> AC_C_P;
    DC_P.clear();
    float EntDCY=0;
    float EntDCC=0;
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, HDQ_OptD::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, HDQ_OptD::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    EntDCY = calHuffmanCodeSize(DC_P);
    DC_P.clear();
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, HDQ_OptD::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, HDQ_OptD::seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, HDQ_OptD::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, HDQ_OptD::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    DC_P.clear();

    for(i=0; i<HDQ_OptD::seq_len_Y; i++){
        HDQ_OptD::RSlst.clear(); HDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], HDQ_OptD::RSlst, HDQ_OptD::IDlst);
        cal_P_from_RSlst(HDQ_OptD::RSlst, AC_Y_P);
    }
    HDQ_OptD::RSlst.clear(); HDQ_OptD::IDlst.clear();
    AC_Y_P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(AC_Y_P);
    AC_Y_P.clear();

    for(i=0; i<HDQ_OptD::seq_len_C; i++){
        HDQ_OptD::RSlst.clear(); HDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], HDQ_OptD::RSlst, HDQ_OptD::IDlst);
        cal_P_from_RSlst(HDQ_OptD::RSlst, AC_C_P);
        HDQ_OptD::RSlst.clear(); HDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], HDQ_OptD::RSlst, HDQ_OptD::IDlst);
        cal_P_from_RSlst(HDQ_OptD::RSlst, AC_C_P);
    }

    AC_C_P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(AC_C_P);
    float BPP=0;
    float file_size = EntACC+EntACY+EntDCC+EntDCY+FLAG_SIZE; // Run_length coding
    BPP = file_size/HDQ_OptD::img_shape_Y[0]/HDQ_OptD::img_shape_Y[1];
    delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    Dequantize(seq_dct_idxs_Y, HDQ_OptD::Q_table_Y, HDQ_OptD::seq_len_Y); //seq_dct_idxs_Y: [][64]
    Dequantize(seq_dct_idxs_Cb, HDQ_OptD::Q_table_C, HDQ_OptD::seq_len_C);
    Dequantize(seq_dct_idxs_Cr, HDQ_OptD::Q_table_C, HDQ_OptD::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Y, blockified_img_Y, HDQ_OptD::seq_len_Y); //seq_dct_idxs_Y: [][8[8]
    seq_2_blockidct(seq_dct_idxs_Cb, blockified_img_Cb, HDQ_OptD::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Cr, blockified_img_Cr, HDQ_OptD::seq_len_C);
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    deblockify(blockified_img_Y,  image[0], HDQ_OptD::img_shape_Y); //seq_dct_idxs_Y: [][], crop here!
    deblockify(blockified_img_Cb, image[1], HDQ_OptD::img_shape_C);
    deblockify(blockified_img_Cr, image[2], HDQ_OptD::img_shape_C);
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    Upsampling(image[1], HDQ_OptD::img_shape_Y, HDQ_OptD::J, HDQ_OptD::a, HDQ_OptD::b);
    Upsampling(image[2], HDQ_OptD::img_shape_Y, HDQ_OptD::J, HDQ_OptD::a, HDQ_OptD::b);
    delete [] DC_idxs_Y; delete [] DC_idxs_Cb; delete [] DC_idxs_Cr;
    return (check * BPP);
}


float HDQ_OptD::SWE_eval(int Sens_index, float Q_table[64],float seq_dct_coefs[][64], float seq_dct_idxs[][64], int seq_len, float d_waterlevel)
{
    Quantize(seq_dct_coefs, seq_dct_idxs, Q_table, seq_len);
    
    if (Sens_index > 0 )
    {
        fast_quatization_CbCr(Sens_index, HDQ_OptD::varianceData_CbCr, seq_dct_idxs, 
                                seq_dct_idxs, d_waterlevel, seq_len);
    }

    Dequantize(seq_dct_idxs, Q_table, seq_len);

    return SWE(HDQ_OptD::Sen_Map, Sens_index,
                            seq_dct_coefs ,
                            seq_dct_idxs, 
                            seq_len);
}

