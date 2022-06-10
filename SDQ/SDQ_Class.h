#include <map>
#include "block.h"
#include <ctime>
#include <chrono>
using namespace std;

const double MIN_Q_VAL = 1.733;
const int TOTAL_KEY = -10011;
class SDQ{
    public:
        // attributes
        double Beta_S;
        double Beta_W;
        double Beta_X;
        //double Sen_Map[3][64] = {0};
        double Q_table_Y[64];
        double Q_table_C[64];
        int seq_len_Y, seq_len_C; // # 8x8 DCT blocks after subsampling
        int n_row;
        int n_col;
        int SmplHstep, SmplVstep;
        int Smplcols, Smplrows;
        BLOCK Block;
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
        double EntPSY, EntPSC;
        vector<int> RSlst;
        vector<int> IDlst;
        void __init__(double eps, double Beta_S, double Beta_W, double Beta_X,
                      double Lmbda, double Sen_Map[3][64], int QF_Y, int QF_C, 
                      int J, int a, int b);
        void Block_to_RSlst(double block[64],vector<int>& RSlst, vector<int>& IDlst);
        void RSlst_to_Block(double DC, vector<int> RSlst,
                            vector<int> IDlst, double block[64]);
        void cal_P_from_RSlst(vector<int> RSlst, map<int, double> & P);
        void norm(map<int, double> P, map<int, double> & ent);
        void cal_ent(map<int, double> & ent);
        void Subsampling(vector<vector<double>>& img);
        void Upsampling(vector<vector<double>>& img);

        void opt_DC(double seq_dct_idxs_Y[][64], double seq_dct_coefs_Y[][64],
                    double seq_dct_idxs_Cr[][64], double seq_dct_coefs_Cr[][64],
                    double seq_dct_idxs_Cb[][64], double seq_dct_coefs_Cb[][64]);
        void opt_RS_Y(double seq_dct_idxs_Y[][64], double seq_dct_coefs_Y[][64]);
        void opt_RS_C(double seq_dct_idxs_Cr[][64], double seq_dct_coefs_Cr[][64],
                      double seq_dct_idxs_Cb[][64], double seq_dct_coefs_Cb[][64]);
        void opt_Q_Y(double seq_dct_idxs_Y[][64], double seq_dct_coefs_Y[][64]);
        void opt_Q_C(double seq_dct_idxs_Cb[][64], double seq_dct_coefs_Cb[][64],
                     double seq_dct_idxs_Cr[][64], double seq_dct_coefs_Cr[][64]);
        void __call__(vector<vector<vector<double>>>& image);
};

void SDQ::__init__(double eps, double Beta_S, double Beta_W, double Beta_X,
                   double Lmbda, double Sen_Map[3][64], int QF_Y, int QF_C, 
                   int J, int a, int b){
    SDQ::RSlst.reserve(64);
    SDQ::IDlst.reserve(64);
    SDQ::RSlst.clear();
    SDQ::IDlst.clear();
    SDQ::Beta_S = Beta_S;
    SDQ::Beta_W = Beta_W;
    SDQ::Beta_X = Beta_X;
    quantizationTable(QF_Y, true, SDQ::Q_table_Y);
    quantizationTable(QF_C, false, SDQ::Q_table_C);
    SDQ::Block.__init__(eps, Beta_S, Beta_W, Beta_X, Lmbda, Sen_Map);
    SDQ::J_Y = 10e10;
    SDQ::J_C = 10e10;
    SDQ::J = J;
    SDQ::a = a;
    SDQ::b = b;
}

void SDQ::__call__(vector<vector<vector<double>>>& image){
    // const std::chrono::time_point<std::chrono::steady_clock> start =
    //     std::chrono::steady_clock::now();
    int i,j,k,l;
    shape(image, SDQ::img_shape_Y);
    SDQ::n_col = SDQ::img_shape_Y[1];
    SDQ::n_row = SDQ::img_shape_Y[0];
    SDQ::seq_len_Y = pad_shape(SDQ::img_shape_Y[0], 8)*pad_shape(SDQ::img_shape_Y[1], 8)/64;

    SDQ::Subsampling(image[1]);
    SDQ::Subsampling(image[2]);
    SDQ::seq_len_C = pad_shape(SDQ::Smplcols, 8)*pad_shape(SDQ::Smplrows, 8)/64;
    auto blockified_img_Y = new double[SDQ::seq_len_Y][8][8];
    auto blockified_img_Cb = new double[SDQ::seq_len_C][8][8];
    auto blockified_img_Cr = new double[SDQ::seq_len_C][8][8];

    auto seq_dct_coefs_Y = new double[SDQ::seq_len_Y][64];
    auto seq_dct_coefs_Cb = new double[SDQ::seq_len_C][64];
    auto seq_dct_coefs_Cr = new double[SDQ::seq_len_C][64];

    auto seq_dct_idxs_Y = new double[SDQ::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new double[SDQ::seq_len_C][64];
    auto seq_dct_idxs_Cr = new double[SDQ::seq_len_C][64];

    blockify(image[0], SDQ::img_shape_Y, blockified_img_Y);
    blockify(image[1], SDQ::img_shape_C, blockified_img_Cb);    
    blockify(image[2], SDQ::img_shape_C, blockified_img_Cr);

    block_2_seqdct(blockified_img_Y, seq_dct_coefs_Y, SDQ::seq_len_Y);
    block_2_seqdct(blockified_img_Cb, seq_dct_coefs_Cb, SDQ::seq_len_C);
    block_2_seqdct(blockified_img_Cr, seq_dct_coefs_Cr, SDQ::seq_len_C);

    Quantize(seq_dct_coefs_Y,seq_dct_idxs_Y, 
             SDQ::Q_table_Y, SDQ::seq_len_Y);
    Quantize(seq_dct_coefs_Cb,seq_dct_idxs_Cb,
             SDQ::Q_table_C,SDQ::seq_len_C);
    Quantize(seq_dct_coefs_Cr,seq_dct_idxs_Cr,
             SDQ::Q_table_C,SDQ::seq_len_C);
    SDQ::opt_DC(seq_dct_idxs_Y,seq_dct_coefs_Y, 
                seq_dct_idxs_Cb,seq_dct_coefs_Cb,
                seq_dct_idxs_Cr,seq_dct_coefs_Cr);
    for(i=0; i<5; i++){
        SDQ::Loss = 0;
        SDQ::Block.state.ent=0;
        SDQ::opt_RS_Y(seq_dct_idxs_Y,seq_dct_coefs_Y);
        SDQ::opt_Q_Y(seq_dct_idxs_Y,seq_dct_coefs_Y);
        // std::cout<<SDQ::Loss<<std::endl;
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int ii, rr,ss;
    // for(map<int, double>::iterator it =SDQ::Block.P.begin(); it != SDQ::Block.P.end(); it++){
    //     ii = it->first;
    //     hash2rs(ii,rr,ss);
    //     cout<<ii<<"::"<<rr<<":"<<ss<<"->"<< it->second<<endl;
    // }
    for(map<int, double>::iterator it =SDQ::Block.P.begin(); it != SDQ::Block.P.end(); it++){
        ii = it->first;
        if(ii!=TOTAL_KEY){
            hash2rs(ii,rr,ss);
            cout<<ii<<"::"<<rr<<":"<<ss<<"->"<< it->second<<endl;
        }
    }cout<<endl;
    // for(map<int, double>::iterator it =SDQ::Block.P.begin(); it != SDQ::Block.P.end(); it++){
    //     ii = it->first;
    //     if(ii!=TOTAL_KEY){
    //     hash2rs(ii,rr,ss);
    //     cout<< it->second<<", ";
    //     }
    // }cout<<endl;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
    EntPSY = SDQ::Block.state.ent;
    for(i=0; i<5; i++){
        SDQ::Loss = 0;
        SDQ::Block.state.ent=0;
        SDQ::opt_RS_C(seq_dct_idxs_Cb,seq_dct_coefs_Cb,
                      seq_dct_idxs_Cr,seq_dct_coefs_Cr);
        SDQ::opt_Q_C(seq_dct_idxs_Cb,seq_dct_coefs_Cb,
                     seq_dct_idxs_Cr,seq_dct_coefs_Cr);
        // std::cout<<SDQ::Loss<<std::endl;
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // for(map<int, double>::iterator it =SDQ::Block.P.begin(); it != SDQ::Block.P.end(); it++){
    //     ii = it->first;
    //     hash2rs(ii,rr,ss);
    //     cout<<ii<<"::"<<rr<<":"<<ss<<"->"<< it->second<<endl;
    // }
    for(map<int, double>::iterator it =SDQ::Block.P.begin(); it != SDQ::Block.P.end(); it++){
        ii = it->first;
        if(ii!=TOTAL_KEY){
            hash2rs(ii,rr,ss);
            cout<<ii<<"::"<<rr<<":"<<ss<<"->"<< it->second<<endl;
        }
    }cout<<endl;
    // for(map<int, double>::iterator it =SDQ::Block.P.begin(); it != SDQ::Block.P.end(); it++){
    //     ii = it->first;
    //     if(ii!=TOTAL_KEY){
    //     hash2rs(ii,rr,ss);
    //     cout<< it->second<<", ";
    //     }
    // }cout<<endl;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
    EntPSC = SDQ::Block.state.ent;
    cout<<"BPP: "<<(EntPSC+EntPSY)/512/512<<endl;
    delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    Dequantize(seq_dct_idxs_Y, SDQ::Q_table_Y, SDQ::seq_len_Y); //seq_dct_idxs_Y: [][64]
    Dequantize(seq_dct_idxs_Cb, SDQ::Q_table_C, SDQ::seq_len_C);
    Dequantize(seq_dct_idxs_Cr, SDQ::Q_table_C, SDQ::seq_len_C);

    seq_2_blockidct(seq_dct_idxs_Y, blockified_img_Y, SDQ::seq_len_Y); //seq_dct_idxs_Y: [][8[8]
    seq_2_blockidct(seq_dct_idxs_Cb, blockified_img_Cb, SDQ::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Cr, blockified_img_Cr, SDQ::seq_len_C);
    
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    deblockify(blockified_img_Y,  image[0], SDQ::img_shape_Y); //seq_dct_idxs_Y: [][], crop here!
    deblockify(blockified_img_Cb, image[1], SDQ::img_shape_C);
    deblockify(blockified_img_Cr, image[2], SDQ::img_shape_C);
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    Upsampling(image[1]);
    Upsampling(image[2]);
    // vector<vector<vector<double>>>padded_img = pad_Vector(image);
    // Huffman code 
    // const auto end = std::chrono::steady_clock::now();
    // std::cout<<(end - start) / 1s << "s.\n";
}
