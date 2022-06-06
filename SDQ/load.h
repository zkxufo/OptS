#include <iostream>
#include <fstream>
#include <string>
using namespace std;
void readData(string inFileName, double data[], int length){
    ifstream inFile;
    inFile.open(inFileName.c_str());
    if (inFile.is_open()) {
        for (int i = 0; i < length; i++){inFile >> data[i];}
        inFile.close();
    }
    else{ cerr << "Can't find input file " << inFileName << endl;}
}

void readData3x3(string inFileName, double data[3][3]){
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

void Tanspose(double arr[][3]){
    double tmp;
    for(int i=0; i<3; i++){
        for(int j=i; j<3; j++){
            tmp = arr[i][j];
            arr[i][j] = arr[j][i];
            arr[j][i] = tmp;
        }
    }
}

void LoadSenMap(string model, double Sen_Map[3][64]){
    if(model != "NoModel"){
        readData("./SenMap/"+model+"_Y_KLT.txt", Sen_Map[0], 64);
        readData("./SenMap/"+model+"_Cb_KLT.txt", Sen_Map[1], 64);
        readData("./SenMap/"+model+"_Cr_KLT.txt", Sen_Map[2], 64);
    }
    else{
        for(int c=0; c<3; c++){
            for(int i=0;i<64;i++){
                Sen_Map[c][i] = 1;
            }
        }
    }
}

void LoadColorConvW(string model, double W_rgb2swx[3][3], double W_swx2rgb[3][3]){
    if(model != "NoModel"){
        readData3x3("./color_conv_W/"+model+"_W_OPT.txt", W_rgb2swx);
        readData3x3("./color_conv_W/"+model+"_W_OPT.txt", W_swx2rgb);
        Tanspose(W_swx2rgb);
    }
}
