#include <math.h>
#include <algorithm>

#define SIZEOF(arr) sizeof(arr) / sizeof(arr[0])
#define FAST_QUANTIZATION 0

const float MINQVALUE = 1.;
const float MAXQVALUE = 255.;

// #define MINQVALUE 1.0f
// #define MAXQVALUE 255.f

using namespace std;


void quantizationTable(int QF, bool Luminance, float Q_Table[64])
{
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


void parameterCal(float varianceData[64], float lambdaData[64],float seq_dct_coefs[][64], int N_block)
{
    float tmp;
    for (int i = 0; i < 64; i++ )
    {
        float variance = 0, mean = 0, lambda = 0;

        for (int j=0;j<N_block; j++)
        {
            mean += seq_dct_coefs[j][i];
        }
        mean /= N_block;

        for (int j=0;j<N_block; j++)
        {
            variance += pow(seq_dct_coefs[j][i] - mean, 2);
        }
        variance /= N_block;
        
        for (int j=0;j<N_block; j++)
        {
            tmp = seq_dct_coefs[j][i] - mean;
            lambda += abs(tmp);
        }
        varianceData[i] = variance;
        lambdaData[i] = lambda/N_block;

        // cout << varianceData[i] << "\t" << lambdaData[i] << "\n";
    }
    // cout << endl;
}



float Cal_DT(float d_waterLevel, float varianceData[], int len)
{
    float DT = 0;
    for (int i = 0; i < len; i++)
    {
        if(d_waterLevel <= varianceData[i]) DT += d_waterLevel;
        else DT += varianceData[i];
    }
    return DT;
}


float Cal_d(float& DT, float varianceData[], int len)
{
    float sum_var = 0;
    for (int i = 0; i <len; i++)
    {
        sum_var += varianceData[i];
    }
    DT = MinMaxClip(DT, 0, sum_var); // clip between 0 and sum_var

    // cout << "sum_var : " << sum_var << endl;

    // bi-section search
    float eps = 1e-4;
    float a = 0;
    // Maximum variance allover 64 Cofficient
    float b = *max_element(varianceData, varianceData + len);

    if (DT == 0) return a;
    // I believe it should be DT== sum_var
    if (DT == sum_var) return b;

    float c = a;
    while ((b-a) >= eps) {
        // Find middle point
        c = (a+b)/2;
        // Check if middle point is root
        if (Cal_DT(c,varianceData, len) == DT)
            break;
        // Decide the side to repeat the steps
        else if ((Cal_DT(c,varianceData,len)-DT)*(Cal_DT(a,varianceData, len)-DT) < 0)
            b = c;
        else
            a = c;
    }
    return c;
}



void OptD_C(float Sen_Map[][64], 
            float varianceData_CbCr[],
            float lambdaData_Cb[], float lambdaData_Cr[], 
            float Q_Table[], float& DT, float& d_waterLevel, int QMAX_C)
{
    auto Dlap_Cb = new float[QMAX_C + 1][64];
    auto Dlap_Cr = new float[QMAX_C + 1][64];
    if (d_waterLevel < 0) d_waterLevel = Cal_d(DT, varianceData_CbCr, 128); // DT will be used only if d_waterLevel < 0
    float si, q_lambda;
    float p1, p2, p3;
    // cout <<  "Q value" << "\t"  << "Variance" << "\t" << "lambda" << "\n";
    lambdaData_Cb[0] = 0.0;
    for (int i = 1; i < 64; i++)
    {
        Dlap_Cb[0][i] = 0.0;
        for (int q = 1; q <= QMAX_C; q++)
        {
            q_lambda = q / lambdaData_Cb[i];
            si = (q - lambdaData_Cb[i]) + ( q / (exp(q_lambda) - 1) );
            p1 = 2 * pow(lambdaData_Cb[i], 2);
            p2 = 2 * q * (lambdaData_Cb[i] + si -0.5 * q);
            p3 = exp(si/lambdaData_Cb[i]) * (1 - exp(-1 * q_lambda));
            Dlap_Cb[q][i] = p1 - (p2/p3);
            // cout << Dlap_Cb[q][i]  << "\n";
            // break;
        }
    }


    lambdaData_Cr[0] = 0.0;
    for (int i = 1; i < 64; i++)
    {
        Dlap_Cr[0][i] = 0.0;
        for (int q = 1; q <= QMAX_C; q++)
        {
            q_lambda = q / lambdaData_Cr[i];
            si = (q - lambdaData_Cr[i]) + ( q / (exp(q_lambda) - 1) );
            p1 = 2 * pow(lambdaData_Cr[i], 2);
            p2 = 2 * q * (lambdaData_Cr[i] + si -0.5 * q);
            p3 = exp(si/lambdaData_Cr[i]) * (1 - exp(-1 * q_lambda));
            Dlap_Cr[q][i] = p1 - (p2/p3);
            // cout << Dlap_Cr[q][i]  << "\n";
            // break;
        }
    }


    for (int i = 0; i < 64; i++)
    {
        if(max(varianceData_CbCr[i], varianceData_CbCr[i+64]) < d_waterLevel)
        {
            // Q_Table[i] = 1e6; // quantize to 0
            Q_Table[i] = MinMaxClip(QMAX_C, MINQVALUE, MAXQVALUE);
            
        }
        else if ((varianceData_CbCr[i] < d_waterLevel) && (varianceData_CbCr[i+64] >= d_waterLevel))
        {
            // cout<<"position: "<<i<<endl;
            if (i == 0) // DC q step
            {
// #if FAST_QUANTIZATION > 0
                Q_Table[i] = MinMaxClip(min(floor(sqrt(12*d_waterLevel / Sen_Map[2][i])), float(QMAX_C))
                                        , MINQVALUE, MAXQVALUE);
// #else
                // Free Search without Qmax
                // Q_Table[i] = MinMaxClip(floor(sqrt(12*d_waterLevel / Sen_Map[2][i]))
                //                         , MINQVALUE, MAXQVALUE);
// #endif
            }
            else
            {

               for (int q = QMAX_C; q >= 1 ; q--)
               {
                    // cout << Dlap[q][i]  << "\t" << d_waterLevel;
                    Q_Table[i] = MINQVALUE;
                    if ((Sen_Map[2][i]* Dlap_Cr[q][i]) <= d_waterLevel)
                    {
                        Q_Table[i] = q;
                        break;
                    }
               }
            }
        }
        else if ((varianceData_CbCr[i] >= d_waterLevel) && (varianceData_CbCr[i+64] < d_waterLevel))
        {
            if (i == 0) // DC q step
            {
// #if FAST_QUANTIZATION > 0
                Q_Table[i] = MinMaxClip(min(floor(sqrt(12*d_waterLevel / Sen_Map[1][i])), float(QMAX_C))
                                        , MINQVALUE, MAXQVALUE);
// #else
                // Free Search without Qmax
                // Q_Table[i] = MinMaxClip(floor(sqrt(12*d_waterLevel / Sen_Map[1][i]))
                //                         , MINQVALUE, MAXQVALUE);
// #endif
            }
            else
            {

               for (int q = QMAX_C; q >= 1 ; q--)
               {
                    // cout << Dlap[q][i]  << "\t" << d_waterLevel;
                    Q_Table[i] = MINQVALUE;
                    if ((Sen_Map[1][i]* Dlap_Cb[q][i]) <= d_waterLevel)
                    {
                        Q_Table[i] = q;
                        break;
                    }
               }
            }
        }
        else // d_waterLevel =< min(varianceData_CbCr[i], varianceData_CbCr[i+64]) 
        {
            if (i == 0) // DC q step
            {
                // Free Search with Qmax
                // Q_Table[i] = MinMaxClip(floor(sqrt(12*d_waterLevel / (Sen_Map[1][i] + Sen_Map[2][i])))
                //                         , MINQVALUE, MAXQVALUE);

                Q_Table[i] = MinMaxClip(min(floor(sqrt(12*d_waterLevel / max(Sen_Map[1][i], Sen_Map[2][i]))), float(QMAX_C))
                                        , MINQVALUE, MAXQVALUE);
            }
            else
            {

               for (int q = QMAX_C; q >= 1 ; q--)
               {
                    // cout << Dlap[q][i]  << "\t" << d_waterLevel;
                    Q_Table[i] = MINQVALUE;
                    if (((Sen_Map[1][i]* Dlap_Cb[q][i]) <= d_waterLevel) && ((Sen_Map[2][i]* Dlap_Cr[q][i]) <= d_waterLevel))
                    {
                        Q_Table[i] = q;
                        break;
                    }
               }
            }      
        }
        // cout <<  Q_Table[i]  << "\t"  << varianceData_CbCr[i]  <<  "\t"  << varianceData_CbCr[i+64] << "\n";
    }


    delete[] Dlap_Cb; delete[] Dlap_Cr;
}

void OptD_Y(float Sen_Map[3][64], float varianceData[64], float lambdaData[64], float Q_Table[64], float& DT, float& d_waterLevel, int QMAX_Y)
{

    auto Dlap = new float[QMAX_Y + 1][64];
    if (d_waterLevel < 0) d_waterLevel = Cal_d(DT, varianceData, 64); // DT will be used only if d_waterLevel < 0
    

    float si, q_lambda;
    float p1, p2, p3;
    // cout <<  "Q value" << "\t"  << "Variance" << "\t"    << "lambda" << "\n";
    lambdaData[0] = 0.0;
    for (int i = 1; i < 64; i++)
    {
        Dlap[0][i] = 0.0;
        for (int q = 1; q <= QMAX_Y; q++)
        {
            q_lambda = q / lambdaData[i];
            si = (q - lambdaData[i]) + ( q / (exp(q_lambda) - 1) );
            p1 = 2 * pow(lambdaData[i], 2);
            p2 = 2 * q * (lambdaData[i] + si -0.5 * q);
            p3 = exp(si/lambdaData[i]) * (1 - exp(-1 * q_lambda));
            Dlap[q][i] = p1 - (p2/p3);
            // cout << Dlap[q][i]  << "\n";
            // break;
        }
    }

    for (int i = 0; i < 64; i++)
    {
        if(varianceData[i] < d_waterLevel)
        {
            // Q_Table[i] = 1e6; // quantize to 0
            Q_Table[i] = MinMaxClip(QMAX_Y, MINQVALUE, MAXQVALUE);
            
        }
        else
        {
            if (i == 0) // DC q step
            {
                // Free Search with Qmax
                // Q_Table[i] = MinMaxClip(floor(sqrt(12*(d_waterLevel/Sen_Map[0][i]))), MINQVALUE, MAXQVALUE);
                Q_Table[i] = MinMaxClip(min(floor(sqrt(12*(d_waterLevel/Sen_Map[0][i]))), float(QMAX_Y)), MINQVALUE, MAXQVALUE);
            }
            else
            {

               for (int q = QMAX_Y; q >= 1 ; q--)
               {
                    // cout << Dlap[q][i]  << "\t" << d_waterLevel;
                    Q_Table[i] = MINQVALUE;
                    if (Dlap[q][i]  <= (d_waterLevel/Sen_Map[0][i]))
                    {
                        Q_Table[i] = q;
                        break;
                    }
               }
            }      
        }
        // cout <<  Q_Table[i]  << "\t"  << varianceData[i] << "\t" << lambdaData[i] << "\n";
    }

    delete[] Dlap;
}


// m += pow((Y1-Y2), 2);

float SWE(float Sen_Map[3][64], int Sens_index,
                float seq_dct_coefs_RAW[][64],
                float seq_dct_coefs[][64],
                int N_block)
{
    float SWE = 0;
    float tmp;
    for (int i = 0; i < 64; i++ )
    {
        for (int j=0;j<N_block; j++)
        {
            tmp = seq_dct_coefs_RAW[j][i]-seq_dct_coefs[j][i];
            SWE += Sen_Map[Sens_index][i]*pow(tmp, 2);

        }
    }

    SWE /= N_block;

    return SWE;

} 


// float SWE_cal_C(float Sen_Map[3][64], 
//             float seq_dct_coefs_Cb_RAW[][64], float seq_dct_coefs_Cr_RAW[][64], 
//             float seq_dct_coefs_Cb[][64], float seq_dct_coefs_Cr[][64],
//             int N_block_C)
// {
//     float SWE_C = 0;
//     float tmp;

//     for (int i = 0; i < 64; i++ )
//     {
//         for (int j=0;j<N_block_C; j++)
//         {
//             tmp = seq_dct_coefs_Cb_RAW[j][i]-seq_dct_coefs_Cb[j][i];
//             SWE_C += Sen_Map[1][i]*pow(tmp, 2);
//             // SWE_C += pow(tmp, 2);

//             tmp = seq_dct_coefs_Cr_RAW[j][i]-seq_dct_coefs_Cr[j][i];
//             SWE_C += Sen_Map[2][i]*pow(tmp, 2);
//             // SWE_C += pow(tmp, 2);

//         }
//     }

//     SWE_C /= N_block_C;

//     return SWE_C;

// } 

void print_Q_table(float Q_Table[64])
{
    for (int i=0 ; i<64; i++)
    {
        cout << Q_Table[i] << "\t";
    }

    cout << endl;
}


void Sens_deepCopy(float src[3][64], float dest[3][64])
{
    for (int c=0 ; c<3; c++)
    {
        for (int i=0 ; i<64; i++)
        {
            dest[c][i] = src[c][i];
        }
    }
}


void Sens_ones(float Sen_Map[3][64])
{
    for (int c=0 ; c<3; c++)
    {
        for (int i=0 ; i<64; i++)
        {
            Sen_Map[c][i] = 1.0;
        }
    }
}



void quantizationTable_OptD_C(float Sen_Map[3][64], float seq_dct_coefs_Cb[][64], float seq_dct_coefs_Cr[][64], 
        float Q_Table[64], int N_block, float& DT, float& d_waterLevel, int QMAX_C, float& max_var_C)
{
    float varianceData_Cb[64], varianceData_Cr[64], varianceData_CbCr[128];
    float lambdaData_Cb[64],  lambdaData_Cr[64];   // No need for DC
    parameterCal(varianceData_Cb, lambdaData_Cb , seq_dct_coefs_Cb ,N_block);
    parameterCal(varianceData_Cr, lambdaData_Cr , seq_dct_coefs_Cr ,N_block);
    float tmp;
    for (int i=0;i<2;i++)
    {
        for(int l=0;l<64;l++)
        {
            if (i == 0) 
            {
                tmp = Sen_Map[1][l] * varianceData_Cb[l];
            }
            else 
            {
                tmp = Sen_Map[2][l] * varianceData_Cr[l];
            }
            varianceData_CbCr[l+i*64] = tmp;
        }
    }

    if(max_var_C  == 0)
    {
        max_var_C = *max_element(varianceData_CbCr, varianceData_CbCr + 128);
    }

    OptD_C(Sen_Map, varianceData_CbCr, lambdaData_Cb, lambdaData_Cr, Q_Table, DT, d_waterLevel, QMAX_C);
}


void quantizationTable_OptD_Y(float Sen_Map[3][64], float seq_dct_coefs[][64], float Q_Table[64], int N_block, 
                float& DT, float& d_waterLevel, int QMAX_Y , float& max_var_Y)
{
    float varianceData[64];
    float lambdaData[64];   // No need for DC
    parameterCal(varianceData, lambdaData , seq_dct_coefs ,N_block);

    for(int i = 0; i <64; i++)
    {
        varianceData[i] = Sen_Map[0][i] * varianceData[i];
    }

    if(max_var_Y  == 0)
    {
        max_var_Y = *max_element(varianceData, varianceData + 64);
    }

    OptD_Y(Sen_Map, varianceData, lambdaData, Q_Table, DT, d_waterLevel,QMAX_Y);

}


