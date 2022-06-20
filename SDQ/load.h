// load.h

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
#include <fstream>
#include <string>
using namespace std;
void readData(string inFileName, float data[], int length){
    ifstream inFile;
    inFile.open(inFileName.c_str());
    if (inFile.is_open()) {
        for (int i = 0; i < length; i++){inFile >> data[i];}
        inFile.close();
    }
    else{ cerr << "Can't find input file " << inFileName << endl;}
}

void readData3x3(string inFileName, float data[3][3]){
    ifstream inFile;
    inFile.open(inFileName.c_str());
    if (inFile.is_open()){
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){inFile >> data[i][j];}
            }
        inFile.close(); 
    }
    else{cerr << "Can't find input file " << inFileName << endl;}
}

void Tanspose(float arr[][3]){
    float tmp;
    for(int i=0; i<3; i++){
        for(int j=i; j<3; j++){
            tmp = arr[i][j];
            arr[i][j] = arr[j][i];
            arr[j][i] = tmp;
        }
    }
}

void LoadSenMap(string model, float Sen_Map[3][64]){
    if(model != "NoModel"){
        readData("./SenMap/"+model+"_Y_KLT.txt", Sen_Map[0], 64);
        readData("./SenMap/"+model+"_Cb_KLT.txt", Sen_Map[1], 64);
        readData("./SenMap/"+model+"_Cr_KLT.txt", Sen_Map[2], 64);
    }
    else{
        for(int c=0; c<3; c++){
            for(int i=0; i<64; i++){
                Sen_Map[c][i] = 1;
            }
        }
    }
}

void LoadColorConvW(string model, float W_rgb2swx[3][3], float W_swx2rgb[3][3]){
    if(model != "NoModel"){
        readData3x3("./color_conv_W/"+model+"_W_OPT.txt", W_rgb2swx);
        readData3x3("./color_conv_W/"+model+"_W_OPT.txt", W_swx2rgb);
        Tanspose(W_swx2rgb);
    }
}
