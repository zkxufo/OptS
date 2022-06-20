// Huffman.h

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
#include <algorithm>
#include "EntUtils.h"
using namespace std;

struct CompareSecond{
    bool operator()(const pair<int, float>& left, const pair<int, float>& right) const{
        return left.second <= right.second;
    }
};
int getMinKey(std::map<int, float> mymap){
  std::pair<int, float> min = *min_element(mymap.begin(), mymap.end(), CompareSecond());
  return min.first;
}

array<int, 256> findHuffmanCodeSize(map<int, float> P){
    int V1;
    int V2;
    float V1Val;
    static array<int, 256> CODESIZE ;
    CODESIZE.fill(0);
    int OTHEERS[256];
    std::fill_n(OTHEERS, 256, -1);
    while (P.size()>=2){
        V1 = getMinKey(P);
        V1Val = P[V1];
        P.erase(V1);
        V2 = getMinKey(P);
        P[V1] = V1Val+P[V2];
        P.erase(V2);
        CODESIZE[V1]++;
        while (OTHEERS[V1]!=-1){
            V1 = OTHEERS[V1];
            CODESIZE[V1]++;
        } 
        OTHEERS[V1]=V2;
        CODESIZE[V2]++;
        while (OTHEERS[V2]!=-1){
            V2 = OTHEERS[V2];
            CODESIZE[V2]++;
        } 
    }
    return CODESIZE;
}

array<int, 33> findNumberOfCode(array<int, 256> CODESIZE){
    static array<int, 33> BITS ;
    BITS.fill(0);
    int Size;
    for(int i=0;i<=256;i++){
        Size = CODESIZE[i];
        if(Size!=0){
            BITS[Size]++;
        }
    }
    return BITS;
}

void adjustBitLengthTo16Bits(array<int, 33> & BITS){
    int i=32,j=0;
    while(1){
        if(BITS[i]>0){
            j=i-1;
            j--;
            while(BITS[j]<=0){j--;}
            BITS[i]=BITS[i]-2;
            BITS[i-1]=BITS[i-1]+1;
            BITS[j+1]=BITS[j+1]+2;
            BITS[j]=BITS[j]-1;
            continue;
        }
        else{
            i--;
            if(i!=16){continue;}
            while(BITS[i]==0){i--;}
            BITS[i]--;
            return;
        }
    }
}

vector<float> sizeForEachCode(array<int, 33> BITS){
    static vector<float> Size;
    Size.clear();
    for(int i=1;i<33;i++){
        for(int j=0;j<BITS[i];j++){
            Size.push_back(i);
        }
    }
    std::reverse(Size.begin(),Size.end());
    return Size;
}

template<typename A, typename B>
std::pair<B,A> flip_pair(const std::pair<A,B> &p){
    return std::pair<B,A>(p.second, p.first);
}
template<typename A, typename B>
std::multimap<B,A> flip_map(const std::map<A,B> &src){
    std::multimap<B,A> dst;
    std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),
                   flip_pair<A,B>);
    return dst;
}

float calHuffmanCodeSize(map<int, float> P){
    float cnt = 0;
    if(P.size()>1){
        array<int, 256> CODESIZE;
        CODESIZE = findHuffmanCodeSize(P);
        array<int, 33> BITS;
        BITS = findNumberOfCode(CODESIZE);
        adjustBitLengthTo16Bits(BITS);
        vector<float> sizeList = sizeForEachCode(BITS);
        std::multimap<float, int> SortedP;
        SortedP = flip_map(P);
        int i=0;
        int r,s;
        float codingsize;
        for(std::multimap<float, int>::iterator it = SortedP.begin(); it != SortedP.end(); it++){
            s = it->second%16;
            codingsize = sizeList[i]+(float)s;
            cnt += codingsize*(it->first);
            i++;
        }
    }
    else{
        cnt = 0;
    }
    return cnt;
}
