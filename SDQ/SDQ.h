#include "SDQ_Class.h"

void SDQ::opt_DC(double seq_dct_idxs_Y[][64],  double seq_dct_coefs_Y[][64],
                 double seq_dct_idxs_Cr[][64], double seq_dct_coefs_Cr[][64],
                 double seq_dct_idxs_Cb[][64], double seq_dct_coefs_Cb[][64]){
    double a = seq_dct_idxs_Y[0][0];
}

void SDQ::opt_RS_Y(double seq_dct_idxs_Y[][64], double seq_dct_coefs_Y[][64]){
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
        // cout<<SDQ::Block.J<<" ";
        SDQ::Loss += SDQ::Block.J;
    }
    // cout<<"Yent: "<<SDQ::Block.state.ent<<" "<<endl<<flush;
}


void SDQ::opt_RS_C(double seq_dct_idxs_Cr[][64], double seq_dct_coefs_Cr[][64],
                   double seq_dct_idxs_Cb[][64], double seq_dct_coefs_Cb[][64]){
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
    ////////////////////////////////////////////////////////////////////////////////
    // for(map<int, double>::iterator it = SDQ::Block.ent.begin(); it != SDQ::Block.ent.end(); it++){
    //     cout<<it->first<<"-->"<<it->second<<endl<<flush;
    // }
////////////////////////////////////////////////////////////////////////////////
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
    // cout<<"Cent: "<<SDQ::Block.state.ent<<" "<<endl<<flush;
}

void SDQ::opt_Q_Y(double seq_dct_idxs_Y[][64], double seq_dct_coefs_Y[][64]){
    double divisor=0;
    double denominator=0;
    double val;
    int i,j;
    for(j=0; j<63; j++){
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

void SDQ::opt_Q_C(double seq_dct_idxs_Cb[][64], double seq_dct_coefs_Cb[][64],
                  double seq_dct_idxs_Cr[][64], double seq_dct_coefs_Cr[][64]){      
    double divisor=0;
    double denominator=0;
    double val;
    int i,j;
    for(j=0; j<63; j++){  
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

void SDQ::Subsampling(vector<vector<double>>& img){
    int Nrows = SDQ::img_shape_Y[0];
    int Ncols = SDQ::img_shape_Y[1];
    int pad_rows = pad_shape(Nrows, SDQ::J);
    int pad_cols = pad_shape(Ncols, SDQ::J);
    SDQ::SmplHstep = floor(SDQ::J/SDQ::a);
    if(b==0){SDQ::SmplVstep = 2;} else{SDQ::SmplVstep = 1;}
    int i;
    int j;
    int hidx;
    int vidx;
    for(i=0, hidx=0; i<pad_rows; i+=SDQ::SmplVstep, hidx++){
        for(j=0, vidx=0;j<pad_cols;j+=SDQ::SmplHstep,vidx++){
            if(j<Ncols && i<Nrows){
                img[hidx][vidx] = img[i][j];
            }
        }
    }
    SDQ::Smplrows = pad_rows/SDQ::SmplVstep;
    SDQ::Smplcols = pad_cols/SDQ::SmplHstep;
    SDQ::img_shape_C[0] = min(Nrows,SDQ::Smplrows);
    SDQ::img_shape_C[1] = min(Ncols,SDQ::Smplcols);
}

void SDQ::Upsampling(vector<vector<double>>& img){
    int Nsampled_row = SDQ::img_shape_Y[0];
    int Nsampled_col = SDQ::img_shape_Y[1];
    int x,y,xidx,yidx;
    int i,j;
    for(i=Nsampled_row-1;i>-1;i--){
        for(j=Nsampled_col-1;j>-1;j--){
            for(x=SDQ::SmplHstep-1; x>-1; x--){
                for(y=SDQ::SmplVstep-1; y>-1; y--){
                    yidx = i*SDQ::SmplVstep+y;
                    xidx = j*SDQ::SmplHstep+x;
                    if(xidx<Nsampled_col && yidx<Nsampled_row){
                        img[yidx][xidx] = img[i][j];
                    }
                }
            }
        }
    }
}
