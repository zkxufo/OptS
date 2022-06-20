// EntUtils.h

// MIT License

// Copyright (c) 2022 deponce(Linfeng Ye), University of Waterloo

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <cstdlib>
#include <map>
#include <vector>
const int TOTAL_KEY = -10011;
void DPCM(float blockified_channel[][64], float differ[], int size){
    /*
    :Fn  DPCM: calculate the Differential Pulse Code Modulation(DPCM)
    :param float blockified_channel[][64]: flatten blockified channel
    :param float differ[]: the DPCM coding sequence
    :param int size: number of blocks in a channel
    :return void:
    */
    int i;
    differ[0] = blockified_channel[0][0];
    for(i=1; i<size; i++){
        differ[i] = blockified_channel[i][0] - blockified_channel[i-1][0];
    }
    return;
}

void cal_P_from_DIFF(float differ[], std::map<int, float> & P, int size){
    /*
    :Fn  cal_P_from_DIFF: Count the size group of DPCM coding sequence in P
    :param float differ[]: the DPCM coding sequence
    :param std::map<int, float> & P: Count the group size, key: size group, value: number of size group 
    :param int size: number of blocks in a channel
    :return void:
    */
    int i;
    float val, sizeGroup;
    for(i=0; i<size; i++){
        val = differ[i];
        sizeGroup = size_group(val, 11, 0);
        if(P.count(sizeGroup)){
            P[sizeGroup] += 1;
        }
        else{
            P[sizeGroup] = 1;
        }
    }
    return;
}

void Block_to_RSlst(float block[64], std::vector<int>& RSlst, std::vector<int>& IDlst){
    /*
    :Fn  Block_to_RSlst: calculate the Run-Size list from a flatten block
    :param float block[64]: flatten block
    :param std::vector<int>& RSlst: Run-Size list
    :param std::vector<int>& IDlst: Index list
    :return void:
    */
    int i,j, R, S, RS;
    int cur_idx=0; // current index
    int RS150 = 0; // counter of run-size pair (15,0)
    for(i=1; i<64; i++){
        R = i-cur_idx-1; // Run
        if(block[i] == 0){
            if(R != 15){continue;}
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
            S = size_group(block[i]); // Size
            RS = rs2hash(R,S);
            RSlst.push_back(RS);
            IDlst.push_back(block[i]);
        }
        cur_idx = i;
    }
    RSlst.push_back(0); // add (0,0), end of block
    return;
}

void RSlst_to_Block(float DC, std::vector<int> RSlst,
                    std::vector<int> IDlst, float block[64]){
    /*
    :Fn  RSlst_to_Block: revert flatten block from the Run-Size list
    :param float block[64]: flatten block
    :param std::vector<int>& RSlst: Run-Size list
    :param std::vector<int>& IDlst: Index list
    :return void:
    */
    std::fill_n(block, 64, 0);
    block[0] = DC;
    int blk_idx = 0;
    int IDidx = 0;
    int r, s;
    for(auto RS: RSlst){
        if(RS!=0){
            hash2rs(RS, r, s);
            blk_idx += r+1;
            if(s!=0){
                block[blk_idx] = IDlst[IDidx];
                IDidx++;
            }
        }
        else{break;}
    }
    return;
}

void norm(std::map<int, float> P, std::map<int, float> & ent){
    /*
    :Fn  norm: normalize the P to a distribution
    :param std::map<int, float> P: count the Run-size pair
    :param std::map<int, float> & ent: distribution
    :return void:
    */
    float total = P[TOTAL_KEY];
    int KEY;
    for(std::map<int, float>::iterator it = P.begin(); it != P.end(); it++){
        KEY = it->first;
        if(KEY!=TOTAL_KEY){
            ent.insert({KEY, 0});
            ent[KEY] = P[KEY]/total;
        }
    }
    return;
}

void cal_ent(std::map<int, float> & ent){
    /*
    :Fn  cal_ent: Entropy rate associated with pair (r,s), i.e, $ent(r,s) = -log_2 P(r,s)+s$
    :param std::map<int, float> P: distribution
    :param std::map<int, float> & ent: Entropy rate
    :return void:
    */
    int R, S;
    int KEY;
    for(std::map<int, float>::iterator it = ent.begin(); it != ent.end(); it++){
        KEY = it->first;
        if(KEY!=TOTAL_KEY){
            hash2rs(KEY, R, S);
            ent[KEY] = (-log2(ent[KEY])+ (float)S);
        }
    }
    return;
}

void cal_P_from_RSlst(std::vector<int> RSlst, std::map<int, float> & P){
    /*
    :Fn  cal_P_from_RSlst: Count the size group of Run-Size list coding sequence in P
    :param std::vector<int> RSlst: Run-Size list
    :param std::map<int, float> & P: Count the group size, key: size group, value: number of size group 
    :return void:
    */
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
    return;
}
