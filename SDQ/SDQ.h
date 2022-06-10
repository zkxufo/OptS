#include "SDQ_Class.h"

void SDQ::Block_to_RSlst(double block[64],vector<int>& RSlst, vector<int>& IDlst){
    //tested
    // convert a flatten block to a Run Size list
    int i,j, r, S, RS;
    int cur_idx=0;
    int RS150 = 0;
    for(i=1; i<64; i++){
        r = i-cur_idx-1;
        if(block[i] == 0){
            if(r != 15){continue;}
            else{
                RS150+=1;
            }
        }
        else{
            if(RS!=0){
                RS = rs2hash(15,0);
                for(j=0; j<RS150; j++){
                    RSlst.push_back(RS);
                }
                RS150=0;}
            S = size_group(block[i]);
            RS = rs2hash(r,S);
            RSlst.push_back(RS);
            IDlst.push_back(block[i]);
        }
        cur_idx = i;
    }
    RSlst.push_back(0);
}

void SDQ::RSlst_to_Block(double DC, vector<int> RSlst,
                         vector<int> IDlst, double block[64]){
    // tested
    // convert a Run Size list to a flatten block
    std::fill_n(block, 64, 0);
    block[0] = DC;
    int blk_idx = 0;
    int ididx = 0;
    int r, s;
    for(auto RS: RSlst){
        if(RS!=0){
            hash2rs(RS, r, s);
            blk_idx = blk_idx+r+1;
            if(s!=0){
                block[blk_idx] = IDlst[ididx];
                ididx += 1;
            }
        }
        else{break;}
    }
}

void SDQ::norm(map<int, double> P, map<int, double> & ent){
    //tested
    double total = P[TOTAL_KEY];
    int KEY;
    // cout<<"total: "<<total<<endl;
    for(map<int, double>::iterator it = P.begin(); it != P.end(); it++){
        KEY = it->first;
        if(KEY!=TOTAL_KEY){
            ent.insert({KEY, P[KEY]/total});
        }
    }
}

void SDQ::cal_ent(map<int, double> & ent){
    //tested
    int R, S;
    int KEY;
    for(map<int, double>::iterator it = ent.begin(); it != ent.end(); it++){
        KEY = it->first;
        if(KEY!=TOTAL_KEY){
            hash2rs(KEY, R, S);
            ent[KEY] = (-log2(ent[KEY])+ (double)S);
        }
    }
}

void SDQ::opt_DC(double seq_dct_idxs_Y[][64],  double seq_dct_coefs_Y[][64],
                 double seq_dct_idxs_Cr[][64], double seq_dct_coefs_Cr[][64],
                 double seq_dct_idxs_Cb[][64], double seq_dct_coefs_Cb[][64]){
    double a = seq_dct_idxs_Y[0][0];
}

void SDQ::cal_P_from_RSlst(vector<int> RSlst, map<int, double> & P){
    int length = RSlst.size(), i;
    int RS;
    P.insert({TOTAL_KEY, 0});
    for(i=0; i<length; i++){
        RS = RSlst[i];
        if(P.count(RS)){
            P[RS] += 1;
        }
        else{
            P.insert({RS, 1});
        }
        P[TOTAL_KEY] += 1;
    }
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
        SDQ::Block_to_RSlst(seq_dct_idxs_Y[i], SDQ::RSlst, SDQ::IDlst);
        SDQ::cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.P);
    }
    SDQ::RSlst.clear(); SDQ::IDlst.clear();
    SDQ::norm(SDQ::Block.P, SDQ::Block.ent);
    SDQ::cal_ent(SDQ::Block.ent);
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
        SDQ::RSlst_to_Block(seq_dct_idxs_Y[i][0], SDQ::RSlst,
                            SDQ::IDlst, seq_dct_idxs_Y[i]);
        SDQ::RSlst.clear();SDQ::IDlst.clear();
        // cout<<SDQ::Block.J<<" ";
        SDQ::Loss += SDQ::Block.J;
    }
    cout<<"Yent: "<<SDQ::Block.state.ent<<" "<<endl<<flush;
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
        SDQ::Block_to_RSlst(seq_dct_idxs_Cr[i], SDQ::RSlst, SDQ::IDlst);
        SDQ::cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.ent);
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        SDQ::Block_to_RSlst(seq_dct_idxs_Cb[i], SDQ::RSlst, SDQ::IDlst);
        SDQ::cal_P_from_RSlst(SDQ::RSlst, SDQ::Block.P);
    }
    SDQ::RSlst.clear(); SDQ::IDlst.clear();
    SDQ::norm(SDQ::Block.P, SDQ::Block.ent);
    SDQ::cal_ent(SDQ::Block.ent);
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
        SDQ::RSlst_to_Block(seq_dct_idxs_Cb[i][0], SDQ::RSlst,
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
        SDQ::RSlst_to_Block(seq_dct_idxs_Cr[i][0], SDQ::RSlst,
                            SDQ::IDlst, seq_dct_idxs_Cr[i]);
        SDQ::RSlst.clear(); SDQ::IDlst.clear();
        SDQ::Loss += SDQ::Block.J;  
    }
    cout<<"Cent: "<<SDQ::Block.state.ent<<" "<<endl<<flush;
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
