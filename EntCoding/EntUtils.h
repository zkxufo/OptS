#include <iostream>
#include <cstdlib>
#include <map>
// #include "../SDQ/utils.h"

void DPCM(double blockified_channel[][64], double differ[], int size){
    int i;
    for(i=1; i<size; i++){
        differ[i] = blockified_channel[i][0] - blockified_channel[i-1][0];
    }
}

void cal_P_from_DIFF(double differ[], map<int, double> & P, int size){
    int i;
    double val;
    for(i=1; i<size; i++){
        val = differ[i];
        if(P.count(val)){
            P[val] += 1;
        }
        else{
            P.insert({val, 1});
        }
    }
}

