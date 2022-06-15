#include <iostream>
#include <cstdlib>
#include <map>
#include <vector>
// #include "../SDQ/utils.h"
const int TOTAL_KEY = -10011;
void DPCM(float blockified_channel[][64], float differ[], int size){
    int i;
    
    differ[0] = blockified_channel[0][0];
    for(i=1; i<size; i++){
        differ[i] = blockified_channel[i][0] - blockified_channel[i-1][0];
    }
}

void cal_P_from_DIFF(float differ[], std::map<int, float> & P, int size){
    int i;
    float val, sizeGroup;
    for(i=1; i<size; i++){
        val = differ[i];
        sizeGroup = size_group(val, 11, 0);
        if(P.count(sizeGroup)){
            P[sizeGroup] += 1;
        }
        else{
            P[sizeGroup] = 1;
            // P.insert({sizeGroup, 1});
        }
    }
}

void Block_to_RSlst(float block[64], std::vector<int>& RSlst, std::vector<int>& IDlst){
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
                RS150++;
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

void RSlst_to_Block(float DC, std::vector<int> RSlst,
                         std::vector<int> IDlst, float block[64]){
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
                ididx++;
            }
        }
        else{break;}
    }
}

void norm(std::map<int, float> P, std::map<int, float> & ent){
    //tested
    float total = P[TOTAL_KEY];
    int KEY;
    for(std::map<int, float>::iterator it = P.begin(); it != P.end(); it++){
        KEY = it->first;
        if(KEY!=TOTAL_KEY){
            ent.insert({KEY, 0});
            ent[KEY] = P[KEY]/total;
        }
    }
}

void cal_ent(std::map<int, float> & ent){
    //tested
    int R, S;
    int KEY;
    for(std::map<int, float>::iterator it = ent.begin(); it != ent.end(); it++){
        KEY = it->first;
        if(KEY!=TOTAL_KEY){
            hash2rs(KEY, R, S);
            ent[KEY] = (-log2(ent[KEY])+ (float)S);
        }
    }
}

void cal_P_from_RSlst(std::vector<int> RSlst, std::map<int, float> & P){
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
