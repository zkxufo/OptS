#include <iostream>
#include <cstdlib>
#include <map>
// #include "../SDQ/utils.h"

void DPCM(double blockified_channel[][64], double differ[], int size){
    int i;
    
    differ[0] = blockified_channel[0][0];
    for(i=1; i<size; i++){
        differ[i] = blockified_channel[i][0] - blockified_channel[i-1][0];
    }
}

void cal_P_from_DIFF(double differ[], map<int, double> & P, int size){
    int i;
    double val, sizeGroup;
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

