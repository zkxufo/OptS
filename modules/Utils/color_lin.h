// Version 1
void rgb2swx(std::vector<std::vector<std::vector<float>>>& rgb_img, float W_matrix[3][3], float bias){
    int i, j, k;
    float r_ch, g_ch, b_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    float ch[3];
    for(i=0; i<nrows; i++){
        for(j=0; j<ncols; j++){
            r_ch = rgb_img[0][i][j];
            g_ch = rgb_img[1][i][j];
            b_ch = rgb_img[2][i][j];
            rgb_img[0][i][j] = (r_ch*W_matrix[0][0]+g_ch*W_matrix[0][1]+b_ch*W_matrix[0][2])-bias*(W_matrix[0][0]+W_matrix[0][1]+W_matrix[0][2]);
            rgb_img[1][i][j] = (r_ch*W_matrix[1][0]+g_ch*W_matrix[1][1]+b_ch*W_matrix[1][2])-bias*(W_matrix[1][0]+W_matrix[1][1]+W_matrix[1][2]);
            rgb_img[2][i][j] = (r_ch*W_matrix[2][0]+g_ch*W_matrix[2][1]+b_ch*W_matrix[2][2])-bias*(W_matrix[2][0]+W_matrix[2][1]+W_matrix[2][2]);
        }
    }
}

void swx2rgb(std::vector<std::vector<std::vector<float>>>& rgb_img, float W_matrix[3][3], float bias){
    int i, j, k;
    float r_ch, g_ch, b_ch;
    float Y_ch, U_ch, V_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    for(j=0; j<nrows; j++){
        for(k=0; k<ncols; k++){
            r_ch=rgb_img[0][j][k]+bias*(W_matrix[0][0]+W_matrix[1][0]+W_matrix[2][0]);
            g_ch=rgb_img[1][j][k]+bias*(W_matrix[0][1]+W_matrix[1][1]+W_matrix[2][1]);
            b_ch=rgb_img[2][j][k]+bias*(W_matrix[0][2]+W_matrix[1][2]+W_matrix[2][2]);
            Y_ch = (r_ch*W_matrix[0][0]+g_ch*W_matrix[0][1]+b_ch*W_matrix[0][2]);
            U_ch = (r_ch*W_matrix[1][0]+g_ch*W_matrix[1][1]+b_ch*W_matrix[1][2]);
            V_ch = (r_ch*W_matrix[2][0]+g_ch*W_matrix[2][1]+b_ch*W_matrix[2][2]);
            rgb_img[0][j][k] = MinMaxClip(Y_ch, MIN_PXL_VAL, MAX_PXL_VAL);
            rgb_img[1][j][k] = MinMaxClip(U_ch, MIN_PXL_VAL, MAX_PXL_VAL);
            rgb_img[2][j][k] = MinMaxClip(V_ch, MIN_PXL_VAL, MAX_PXL_VAL);
        }
    }
}

// Version 2

void rgb2swx(std::vector<std::vector<std::vector<float>>>& rgb_img, float W_matrix[3][3], float bias){
    int i, j, k;
    float r_ch, g_ch, b_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    float ch[3];
    for(i=0; i<nrows; i++){
        for(j=0; j<ncols; j++){
            r_ch = rgb_img[0][i][j];
            g_ch = rgb_img[1][i][j];
            b_ch = rgb_img[2][i][j];
            rgb_img[0][i][j] = (r_ch*W_matrix[0][0]+g_ch*W_matrix[0][1]+b_ch*W_matrix[0][2])/(W_matrix[0][0]+W_matrix[0][1]+W_matrix[0][2])-bias;
            rgb_img[1][i][j] = (r_ch*W_matrix[1][0]+g_ch*W_matrix[1][1]+b_ch*W_matrix[1][2])/(W_matrix[1][0]+W_matrix[1][1]+W_matrix[1][2])-bias;
            rgb_img[2][i][j] = (r_ch*W_matrix[2][0]+g_ch*W_matrix[2][1]+b_ch*W_matrix[2][2])/(W_matrix[2][0]+W_matrix[2][1]+W_matrix[2][2])-bias;
        }
    }
}

void swx2rgb(std::vector<std::vector<std::vector<float>>>& rgb_img, float W_matrix[3][3], float bias){
    int i, j, k;
    float r_ch, g_ch, b_ch;
    float Y_ch, U_ch, V_ch;
    int nrows = rgb_img[0].size();
    int ncols = rgb_img[0][0].size();
    for(j=0; j<nrows; j++){
        for(k=0; k<ncols; k++){
            r_ch=(rgb_img[0][j][k]+bias)*(W_matrix[0][0]+W_matrix[1][0]+W_matrix[2][0]);
            g_ch=(rgb_img[1][j][k]+bias)*(W_matrix[0][1]+W_matrix[1][1]+W_matrix[2][1]);
            b_ch=(rgb_img[2][j][k]+bias)*(W_matrix[0][2]+W_matrix[1][2]+W_matrix[2][2]);
            Y_ch = (r_ch*W_matrix[0][0]+g_ch*W_matrix[0][1]+b_ch*W_matrix[0][2]);
            U_ch = (r_ch*W_matrix[1][0]+g_ch*W_matrix[1][1]+b_ch*W_matrix[1][2]);
            V_ch = (r_ch*W_matrix[2][0]+g_ch*W_matrix[2][1]+b_ch*W_matrix[2][2]);
            rgb_img[0][j][k] = MinMaxClip(Y_ch, MIN_PXL_VAL, MAX_PXL_VAL);
            rgb_img[1][j][k] = MinMaxClip(U_ch, MIN_PXL_VAL, MAX_PXL_VAL);
            rgb_img[2][j][k] = MinMaxClip(V_ch, MIN_PXL_VAL, MAX_PXL_VAL);
        }
    }
}



