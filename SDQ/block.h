#include <math.h>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

const double INIT_LOSS = 1e30;
const double  ZERO = 0;
using namespace std;

class BLOCK{
    public:
        // attributes
        double eps;
        double Q_table[64];
        const int ID_max_abs[10] = {1, 3, 7, 15,31, 63, 127, 255, 511, 1023};
        const int ID_min_abs[10] = {1, 2, 4, 8, 16, 32, 64,  128, 256, 512 };
        node state;
        map<int, double> ent;
        map<int, double> P;
        double Sen_Map[3][64]; //'S', 'W', 'X'
        double J = INIT_LOSS;
        double Lmbda;
        int Sen_Map_Idx =0;
        char channel; //'S', 'W', 'X'
        // methods
        double cal_ent(int r, int size);
        void cal_RS(double C[64], double ind[64], 
                    vector<int>& RSlst,
                    vector<int>& IDlst);
        double eob_cost(int i, double CumC[64]);
        double dist(int r,int i, double CumC[64]);
        void __init__(double eps, double Beta_S, double  Beta_W,
                     double Beta_X, double Lmbda, double Sen_Map[3][64]);
        void set_Q_table(double Q_table[64]);
        void set_channel(char channel);
};

double BLOCK::cal_ent(int r, int size){
    int hash_val = rs2hash(r,size);
    double ent_val;
    if(BLOCK::ent.count(hash_val)){
        ent_val = BLOCK::ent[hash_val];
    }
    else{
        ent_val = INIT_LOSS;
    }
    return ent_val;
}

void BLOCK::__init__(double eps, double Beta_S, double  Beta_W,
                     double Beta_X, double Lmbda, double Sen_Map[3][64]){
    int i;
    BLOCK::eps = eps;
    std::fill_n(BLOCK::state.cost,64,0);
    for(i=0; i<64; i++){
        BLOCK::Sen_Map[0][i] = Beta_S*Sen_Map[0][i];
        BLOCK::Sen_Map[1][i] = Beta_W*Sen_Map[1][i];
        BLOCK::Sen_Map[2][i] = Beta_X*Sen_Map[2][i];
        //cout<<BLOCK::Sen_Map[0][i]<<" "<<BLOCK::Sen_Map[1][i]<<" "<<BLOCK::Sen_Map[2][i]<<" "<<endl;
    }
    BLOCK::Lmbda = Lmbda;
    cout<<BLOCK::Lmbda<<endl;
}

void BLOCK::set_Q_table(double Q_table[64]){
    int i;
    for(i=0; i<64; i++){BLOCK::Q_table[i] = Q_table[i];}
}

void BLOCK::set_channel(char channel){
    BLOCK::channel = channel;
    switch (channel){
        case 'S':{BLOCK::Sen_Map_Idx = 0;break;}
        case 'W':{BLOCK::Sen_Map_Idx = 1;break;}
        case 'X':{BLOCK::Sen_Map_Idx = 2;break;}
        default: break;
    }
}

double BLOCK::dist(int r,int i, double CumC[64]){
    if(r==0){
        return 0.;
    }
    else{
        return CumC[i-1]-CumC[i-r-1];
    }
}

double BLOCK::eob_cost(int i, double CumC[64]){
    double cost;
    if (i >= 63){
        cost=0;
    }
    else{
        cost=CumC[63]-CumC[i]+ cal_ent(0,0);
    }
    return cost;
}

void BLOCK::cal_RS(double C[64], double ind[64], 
                   vector<int>& RSlst,
                   vector<int>& IDlst){
    int i,r,s;
    double Sign;
    double cumsum_C[64];cumsum_C[0]=0;
    int s_idx = 0;
    int ID[3][64];
    int S[64];
    double D[3][64];
    int curnt_gap;
    for(i=1; i<64; i++){
        ind[i] = round(C[i]/BLOCK::Q_table[i]);
        S[i]=size_group(ind[i], 10, 0); // S the size group of 64 DCT coefficients in zig-zag order
    }
    cumsum(C, cumsum_C, BLOCK::Sen_Map[BLOCK::Sen_Map_Idx]); 
    int upbnd = find_the_last_non_zero(ind);
    for(i=1; i<=upbnd; i++){
        Sign = sign(C[i]);
        switch (S[i]){
            case 0 ... 1:{
                ID[0][i] = Sign;   
                ID[1][i] = 2*Sign; 
                ID[2][i] = 4*Sign;
                break;
            }
            case 2 ... 9:{
                ID[0][i] = ID_max_abs[S[i]-2] *Sign;
                ID[1][i] = ind[i];
                ID[2][i] = ID_min_abs[S[i]] *Sign;
                break;
            }
            default:{
                ID[0][i] = 255*Sign; // size group 8
                ID[1][i] = 511*Sign; // size group 9
                ID[2][i] = ind[i];   // size group 10
                //cout<<ID[0][i]<<" "<<ID[1][i]<<" "<<ID[2][i]<<" "<<endl;
                break;
            }
        }
        for (s=0; s<3; s++){// s is the size group
            D[s][i] = pow(C[i]-ID[s][i]*Q_table[i], 2);
        }
    }
    double J_lst[63]= {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                       0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                       0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
                       0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}; 
    double curr_minicost;
    int i_nl, size;
    double dist_inc, mean_square_dist;
    for(i=1; i<=upbnd; i++){
        curr_minicost = INIT_LOSS;
        if(i < 16){i_nl=i;}else{i_nl=16;}
        for(r=0; r<i_nl; r++){
            mean_square_dist = BLOCK::dist(r, i, cumsum_C);
            if (mean_square_dist>curr_minicost){
                continue;
            }
            switch (S[i]){
                case 0 ... 1:{
                    for(s_idx=0; s_idx<3; s_idx++){ //s_idx: 0 1 2
                        size=s_idx+1;               //size : 1 2 3
                        dist_inc = BLOCK::Sen_Map[BLOCK::Sen_Map_Idx][i]*(D[s_idx][i]) + mean_square_dist;
                        J = state.cost[i-r-1] + dist_inc + BLOCK::Lmbda*cal_ent(r,size);
                        if (J<curr_minicost){
                            curr_minicost = J;
                            state.rs[i] = rs2hash(r, size);
                            state.cost[i] = J;
                            state.ID[i] = ID[s_idx][i];
                        }
                    }
                    break; 
                }
                case 2 ... 9:{
                    for(s_idx=0; s_idx<3; s_idx++){ //s_idx: 0   1  2
                        size = S[i]-1+s_idx;        //size : S-1 S  S+1
                        dist_inc = BLOCK::Sen_Map[BLOCK::Sen_Map_Idx][i]*(D[s_idx][i])+mean_square_dist;
                        J = state.cost[i-r-1] + dist_inc + BLOCK::Lmbda*cal_ent(r,size);
                        if (J<curr_minicost){
                            curr_minicost = J;
                            state.rs[i] = rs2hash(r,size);
                            state.cost[i] = J;
                            state.ID[i] = ID[s_idx][i];
                        }
                    }
                    break;
                }
                default:{
                    for(s_idx=0; s_idx<3; s_idx++){ //s_idx: 0 1 2 
                        size = s_idx+8;             //size : 8 9 10
                        dist_inc = BLOCK::Sen_Map[BLOCK::Sen_Map_Idx][i]*(D[s_idx][i])+mean_square_dist;
                        J = state.cost[i-r-1] + dist_inc + BLOCK::Lmbda*BLOCK::cal_ent(r,size);
                        if (J<curr_minicost){
                            curr_minicost = J;
                            state.rs[i] = rs2hash(r,size);
                            state.cost[i] = J;
                            state.ID[i] = ID[s_idx][i];
                        }
                    }
                    break;
                }
            }
        }
        // consider the special transition (15,0) for state i (i\geq 16)
        if ((16 <= i)&&(i <= 62)){
            dist_inc = BLOCK::dist(15, i, cumsum_C) + BLOCK::Sen_Map[BLOCK::Sen_Map_Idx][i]*(pow(C[i],2));
            J = state.cost[i-16] + dist_inc + BLOCK::Lmbda*BLOCK::cal_ent(15,0);
            if(J<=curr_minicost){
                curr_minicost = J;
                state.rs[i] = rs2hash(15,0);
                state.cost[i] = J;
            }
        }
        // cost of each node
        // J_lst: [1,2,3 ... 63]
    }
    int curr_idx;
    // step 3 find the optimal path for current block
    state.cost[63] = eob_cost(0, cumsum_C);
    for(i=0; i<=63; i++){
        J_lst[i]=state.cost[i+1]+eob_cost(i+1, cumsum_C);
        if(J_lst[i]<state.cost[63]){
            state.cost[63] = J_lst[i];
        }
    }
    BLOCK::J = state.cost[63];
    curr_idx = 63;
    int rs;
    while (curr_idx>=1){
        rs = state.rs[curr_idx];
        hash2rs(rs, r, s);
        if (rs!=0){
            state.ent += BLOCK::cal_ent(r,s);
            RSlst.push_back(state.rs[curr_idx]);
            if(s!=0){
                IDlst.push_back(state.ID[curr_idx]);
            }
        }
        curnt_gap = r;
        curr_idx = curr_idx-curnt_gap-1;
    }
    std::reverse(RSlst.begin(),RSlst.end());
    std::reverse(IDlst.begin(),IDlst.end());
    RSlst.push_back(0);
}
