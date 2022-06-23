// Q_table.h

// MIT License

// Copyright (c) 2022 Ahmed Hussein Salamah, deponce(Linfeng Ye), Kaixiang Zheng, University of Waterloo

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

#include <math.h>
using namespace std;
const float MINQVALUE = 1.;
const float MAXQVALUE = 255.;

void quantizationTable(int QF, bool Luminance, float Q_Table[64]){
    QF = max(min(QF, 100),0);
    if(QF==0){
        QF=1;
    }
    float quantizationTableData_Y[64]={16.,  11.,  12.,  14.,  12.,  10.,  16.,  14.,  13.,  14.,  18.,  17.,
                                        16.,  19.,  24.,  40.,  26.,  24.,  22.,  22.,  24.,  49.,  35.,  37.,
                                        29.,  40.,  58.,  51.,  61.,  60.,  57.,  51.,  56.,  55.,  64.,  72.,
                                        92.,  78.,  64.,  68.,  87.,  69.,  55.,  56.,  80., 109.,  81.,  87.,
                                        95.,  98., 103., 104., 103.,  62.,  77., 113., 121., 112., 100., 120.,
                                        92., 101., 103.,  99.};
    float quantizationTableData_C[64]={17., 18., 18., 24., 21., 24., 47., 26., 26., 47., 99., 66., 56., 66.,
                                        99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99.,
                                        99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99.,
                                        99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99.,
                                        99., 99., 99., 99., 99., 99., 99., 99.};
    float S;
    float q;              
    if(QF<50){
        S = 5000/QF;
    }
    else{
        S = 200-2*QF;
    }
    if (Luminance == true){
        for(int i=0; i<64; i++){
            q = (50+S*quantizationTableData_Y[i])/100;
            Q_Table[i] = MinMaxClip(floor(q), MINQVALUE, MAXQVALUE);
        }
    }
    else{
        for(int i=0; i<64; i++){
            q = (50+S*quantizationTableData_C[i])/100;
            Q_Table[i] = MinMaxClip(floor(q), MINQVALUE, MAXQVALUE);
        }
    }
}