#include "SDQ_Class.h"

void SDQ::opt_DC(float seq_dct_idxs_Y[][64],  float seq_dct_coefs_Y[][64],
                 float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64],
                 float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]){
    float a = seq_dct_idxs_Y[0][0];
}

void SDQ::opt_RS_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    int i;
    // C channel
    std::fill_n(SDQ::Block.state.ID, 64,0);
    std::fill_n(SDQ::Block.state.rs, 64,0);
    SDQ::Block.ent.clear();
    SDQ::Block.P.clear();
    // initialize Py0
    for(i=0; i<SDQ::seq_len_Y; i++){
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], SDQ::RSlst, SDQ::IDlst);
        cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.P);
    }
    SDQ::RSlst.clear(); SDQ::IDlst.clear();
    norm(SDQ::Block.P, SDQ::Block.ent);
    cal_ent(SDQ::Block.ent);
    SDQ::Block.set_channel('S');
    SDQ::Block.set_Q_table(SDQ::Q_table_Y); 
    SDQ::Loss = 0;
    SDQ::Block.state.ent=0;
    for(i=0; i<SDQ::seq_len_Y; i++){
        std::fill_n(SDQ::Block.state.ID, 64,0);
        std::fill_n(SDQ::Block.state.rs, 64,0);
        SDQ::Block.cal_RS(seq_dct_coefs_Y[i],
                          seq_dct_idxs_Y[i],
                          SDQ::RSlst, SDQ::IDlst);
        RSlst_to_Block(seq_dct_idxs_Y[i][0], SDQ::RSlst,
                            SDQ::IDlst, seq_dct_idxs_Y[i]);
        SDQ::RSlst.clear();SDQ::IDlst.clear();
        SDQ::Loss += SDQ::Block.J;
    }
}


void SDQ::opt_RS_C(float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64],
                   float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]){
    int i;
    // C channel
    SDQ::Block.ent.clear();
    SDQ::Block.P.clear();
    std::fill_n(SDQ::Block.state.ID, 64,0);
    std::fill_n(SDQ::Block.state.rs, 64,0);
    // initialize Pc0
    for(i=0; i<SDQ::seq_len_C; i++){
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], SDQ::RSlst, SDQ::IDlst);
        cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.ent);
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], SDQ::RSlst, SDQ::IDlst);
        cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.P);
    }
    SDQ::RSlst.clear(); SDQ::IDlst.clear();
    norm(SDQ::Block.P, SDQ::Block.ent);
    cal_ent(SDQ::Block.ent);
    SDQ::Block.set_channel('W');
    SDQ::Block.set_Q_table(SDQ::Q_table_C);
    SDQ::Loss = 0;
    SDQ::Block.state.ent=0;
    for(i=0; i<SDQ::seq_len_C; i++){
        std::fill_n(SDQ::Block.state.ID, 64,0);
        std::fill_n(SDQ::Block.state.rs, 64,0);
        SDQ::Block.cal_RS(seq_dct_coefs_Cb[i],
                          seq_dct_idxs_Cb[i],
                          SDQ::RSlst, SDQ::IDlst);
        RSlst_to_Block(seq_dct_idxs_Cb[i][0], SDQ::RSlst,
                            SDQ::IDlst, seq_dct_idxs_Cb[i]);
        SDQ::RSlst.clear();SDQ::IDlst.clear();
        SDQ::Loss += SDQ::Block.J;
    }
    SDQ::Block.set_channel('X');
    for(i=0; i<SDQ::seq_len_C; i++){
        std::fill_n(SDQ::Block.state.ID, 64,0);
        std::fill_n(SDQ::Block.state.rs, 64,0);
        SDQ::Block.cal_RS(seq_dct_coefs_Cr[i],
                          seq_dct_idxs_Cr[i],
                          SDQ::RSlst, SDQ::IDlst);
        RSlst_to_Block(seq_dct_idxs_Cr[i][0], SDQ::RSlst,
                            SDQ::IDlst, seq_dct_idxs_Cr[i]);
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        SDQ::Loss += SDQ::Block.J;  
    }
}

void SDQ::opt_Q_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    float divisor=0;
    float denominator=0;
    float val;
    int i,j;
    //TODO: start with 1
    for(j=1; j<63; j++){
        for(i=0; i<SDQ::seq_len_Y; i++){
            divisor += seq_dct_coefs_Y[i][j+1]*seq_dct_idxs_Y[i][j+1];
            denominator += pow(seq_dct_idxs_Y[i][j+1],2);
        }
        if(denominator != 0){
            val = divisor/denominator;
            if (val<MIN_Q_VAL){val = MIN_Q_VAL;}
            SDQ::Q_table_Y[j+1] = val;
        }
        divisor=0;denominator=0; 
    }
}

void SDQ::opt_Q_C(float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64],
                  float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64]){      
    float divisor=0;
    float denominator=0;
    float val;
    int i,j;
    //TODO: start with 1
    for(j=1; j<63; j++){  
        for(i=0; i<SDQ::seq_len_C; i++){
            divisor+=seq_dct_coefs_Cb[i][j+1]*seq_dct_idxs_Cb[i][j+1];
            divisor += seq_dct_coefs_Cr[i][j+1]*seq_dct_idxs_Cr[i][j+1];
            denominator += pow(seq_dct_idxs_Cb[i][j+1],2);
            denominator += pow(seq_dct_idxs_Cr[i][j+1],2);
        }
        if(denominator != 0){
            val = divisor/denominator;
            if (val<MIN_Q_VAL){val = MIN_Q_VAL;}
            SDQ::Q_table_C[j+1] = val;
        }    
        divisor=0;denominator=0; 
    }
}
