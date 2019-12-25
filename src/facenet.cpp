//
// Created by ChrisKong on 2019/10/17.
//

#include "facenet.h"

facenet::facenet() {

}

facenet::~facenet() {

}

void facenet::printData(pBox *in) {
    for (long i = 0; i < in->height * in->width * in->channel; ++i) {
//        if (in->pdata[i] != 0)
        printf("%f\n", in->pdata[i]);
    }
    cout << "printData" << endl;
}

void facenet::run(Mat &image, vector<mydataFmt> &o, int count) {
    cout << "=====This is No." + to_string(count) + " Picture=====" << endl;
    pBox *output = new pBox;
    pBox *input;
    Stem(image, output);
//    printData(output);
//    return;
    cout << "Stem Finally" << endl;
    input = output;
    output = new pBox;
    for (int i = 0; i < 5; ++i) {
//        model_128/block35_1_list.txt
        string filepath = "../model_" + to_string(Num) + "/block35_" + to_string((i + 1)) + "_list.txt";
        Inception_resnet_A(input, output, filepath, 0.17);
        input = output;
        output = new pBox;
    }
    cout << "Inception_resnet_A Finally" << endl;
    Reduction_A(input, output);
    cout << "Reduction_A Finally" << endl;
    input = output;
//    freepBox(output);
    output = new pBox;
    for (int j = 0; j < 10; ++j) {
//        model_128/block17_1_list.txt
        string filepath = "../model_" + to_string(Num) + "/block17_" + to_string((j + 1)) + "_list.txt";
        Inception_resnet_B(input, output, filepath, 0.1);
        input = output;
        output = new pBox;
    }
    cout << "Inception_resnet_B Finally" << endl;
    Reduction_B(input, output);
    cout << "Reduciotn_B Finally" << endl;
    input = output;
//    freepBox(output);
    output = new pBox;
    for (int k = 0; k < 5; ++k) {
//        model_128/block8_1_list.txt
        string filepath = "../model_" + to_string(Num) + "/block8_" + to_string((k + 1)) + "_list.txt";
//        cout << filepath << endl;
        Inception_resnet_C(input, output, filepath, 0.2);
        input = output;
//        freepBox(output);
        output = new pBox;
    }
    cout << "Inception_resnet_C Finally" << endl;
    Inception_resnet_C_None(input, output, "../model_" + to_string(Num) + "/Block8_list.txt");
    cout << "Inception_resnet_C_None Finally" << endl;
    input = output;
//    freepBox(output);
    output = new pBox;
    AveragePooling(input, output);
    cout << "AveragePooling Finally" << endl;
    input = output;
//    freepBox(output);
    output = new pBox;
    Flatten(input, output);
    cout << "Flatten Finally" << endl;
    input = output;
    output = new pBox;
    fully_connect(input, output, "../model_" + to_string(Num) + "/Bottleneck_list.txt");
    cout << "Fully_Connect Finally" << endl;
    mydataFmt sq = 0, sum = 0;
    for (int i = 0; i < Num; ++i) {
        sq = pow(output->pdata[i], 2);
        sum += sq;
    }
    mydataFmt divisor = 0;
    if (sum < 1e-10) {
        divisor = sqrt(1e-10);
    } else {
        divisor = sqrt(sum);
    }
    for (int j = 0; j < Num; ++j) {
//        o[j] = output->pdata[j] / divisor;
        o.push_back(output->pdata[j] / divisor);
    }
//    memcpy(o, output->pdata, Num * sizeof(mydataFmt));
    freepBox(output);
}

void facenet::Stem(Mat &image, pBox *output) {
    pBox *rgb = new pBox;
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;

    struct pBox *pooling1_out = new pBox;

    struct BN *conv1_var = new BN;
    struct BN *conv1_mean = new BN;
    struct BN *conv1_beta = new BN;

    struct BN *conv2_var = new BN;
    struct BN *conv2_mean = new BN;
    struct BN *conv2_beta = new BN;

    struct BN *conv3_var = new BN;
    struct BN *conv3_mean = new BN;
    struct BN *conv3_beta = new BN;

    struct BN *conv4_var = new BN;
    struct BN *conv4_mean = new BN;
    struct BN *conv4_beta = new BN;

    struct BN *conv5_var = new BN;
    struct BN *conv5_mean = new BN;
    struct BN *conv5_beta = new BN;

    struct BN *conv6_var = new BN;
    struct BN *conv6_mean = new BN;
    struct BN *conv6_beta = new BN;

    long conv1 = initConvAndFc(conv1_wb, 32, 3, 3, 2, 0);
    initBN(conv1_var, conv1_mean, conv1_beta, 32);
    long conv2 = initConvAndFc(conv2_wb, 32, 32, 3, 1, 0);
    initBN(conv2_var, conv2_mean, conv2_beta, 32);
    long conv3 = initConvAndFc(conv3_wb, 64, 32, 3, 1, 1);
    initBN(conv3_var, conv3_mean, conv3_beta, 64);
    long conv4 = initConvAndFc(conv4_wb, 80, 64, 1, 1, 0);
    initBN(conv4_var, conv4_mean, conv4_beta, 80);
    long conv5 = initConvAndFc(conv5_wb, 192, 80, 3, 1, 0);
    initBN(conv5_var, conv5_mean, conv5_beta, 192);
    long conv6 = initConvAndFc(conv6_wb, 256, 192, 3, 2, 0);
    initBN(conv6_var, conv6_mean, conv6_beta, 256);

    long dataNumber[24] = {conv1, 32, 32, 32, conv2, 32, 32, 32, conv3, 64, 64, 64, conv4, 80, 80, 80, conv5, 192, 192,
                           192, conv6, 256, 256, 256};

    mydataFmt *pointTeam[24] = {conv1_wb->pdata, conv1_var->pdata, conv1_mean->pdata, conv1_beta->pdata, \
                            conv2_wb->pdata, conv2_var->pdata, conv2_mean->pdata, conv2_beta->pdata, \
                            conv3_wb->pdata, conv3_var->pdata, conv3_mean->pdata, conv3_beta->pdata, \
                            conv4_wb->pdata, conv4_var->pdata, conv4_mean->pdata, conv4_beta->pdata, \
                            conv5_wb->pdata, conv5_var->pdata, conv5_mean->pdata, conv5_beta->pdata, \
                            conv6_wb->pdata, conv6_var->pdata, conv6_mean->pdata, conv6_beta->pdata};
    string filename = "../model_" + to_string(Num) + "/stem_list.txt";
    readData(filename, dataNumber, pointTeam, 24);

//    if (firstFlag) {
    image2MatrixInit(image, rgb);
    image2Matrix(image, rgb, 1);

    convolutionInit(conv1_wb, rgb, conv1_out);
    //conv1 149 x 149 x 32
    convolution(conv1_wb, rgb, conv1_out);
//    printData(conv1_out);
    BatchNorm(conv1_out, conv1_var, conv1_mean, conv1_beta);
//    printData(conv1_out);
    relu(conv1_out, conv1_wb->pbias);

    convolutionInit(conv2_wb, conv1_out, conv2_out);
    //conv2 147 x 147 x 32
    convolution(conv2_wb, conv1_out, conv2_out);
    BatchNorm(conv2_out, conv2_var, conv2_mean, conv2_beta);
    relu(conv2_out, conv2_wb->pbias);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 147 x 147 x 64
    convolution(conv3_wb, conv2_out, conv3_out);
    BatchNorm(conv3_out, conv3_var, conv3_mean, conv3_beta);
    relu(conv3_out, conv3_wb->pbias);

    maxPoolingInit(conv3_out, pooling1_out, 3, 2);
    //maxPooling 73 x 73 x 64
    maxPooling(conv3_out, pooling1_out, 3, 2);

    convolutionInit(conv4_wb, pooling1_out, conv4_out);
    //conv4 73 x 73 x 80
    convolution(conv4_wb, pooling1_out, conv4_out);
    BatchNorm(conv4_out, conv4_var, conv4_mean, conv4_beta);
    relu(conv4_out, conv4_wb->pbias);

    convolutionInit(conv5_wb, conv4_out, conv5_out);
    //conv5 71 x 71 x 192
    convolution(conv5_wb, conv4_out, conv5_out);
    BatchNorm(conv5_out, conv5_var, conv5_mean, conv5_beta);
    relu(conv5_out, conv5_wb->pbias);


    convolutionInit(conv6_wb, conv5_out, output);
    //conv6 35 x 35 x 256
    convolution(conv6_wb, conv5_out, output);
    BatchNorm(output, conv6_var, conv6_mean, conv6_beta);
    relu(output, conv6_wb->pbias);
//        firstFlag = false;
//    }

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(pooling1_out);

    freepBox(rgb);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);

    freeBN(conv1_var);
    freeBN(conv1_mean);
    freeBN(conv1_beta);

    freeBN(conv2_var);
    freeBN(conv2_mean);
    freeBN(conv2_beta);

    freeBN(conv3_var);
    freeBN(conv3_mean);
    freeBN(conv3_beta);

    freeBN(conv4_var);
    freeBN(conv4_mean);
    freeBN(conv4_beta);

    freeBN(conv5_var);
    freeBN(conv5_mean);
    freeBN(conv5_beta);

    freeBN(conv6_var);
    freeBN(conv6_mean);
    freeBN(conv6_beta);
}

void facenet::Inception_resnet_A(pBox *input, pBox *output, string filepath, float scale) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;
    pBox *conv7_out = new pBox;
    pBox *conv8_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;
    struct Weight *conv7_wb = new Weight;
    struct Weight *conv8_wb = new Weight;

    struct BN *conv1_var = new BN;
    struct BN *conv1_mean = new BN;
    struct BN *conv1_beta = new BN;

    struct BN *conv2_var = new BN;
    struct BN *conv2_mean = new BN;
    struct BN *conv2_beta = new BN;

    struct BN *conv3_var = new BN;
    struct BN *conv3_mean = new BN;
    struct BN *conv3_beta = new BN;

    struct BN *conv4_var = new BN;
    struct BN *conv4_mean = new BN;
    struct BN *conv4_beta = new BN;

    struct BN *conv5_var = new BN;
    struct BN *conv5_mean = new BN;
    struct BN *conv5_beta = new BN;

    struct BN *conv6_var = new BN;
    struct BN *conv6_mean = new BN;
    struct BN *conv6_beta = new BN;


    long conv1 = initConvAndFc(conv1_wb, 32, 256, 1, 1, 0);
    initBN(conv1_var, conv1_mean, conv1_beta, 32);

    long conv2 = initConvAndFc(conv2_wb, 32, 256, 1, 1, 0);
    initBN(conv2_var, conv2_mean, conv2_beta, 32);
    long conv3 = initConvAndFc(conv3_wb, 32, 32, 3, 1, 1);
    initBN(conv3_var, conv3_mean, conv3_beta, 32);

    long conv4 = initConvAndFc(conv4_wb, 32, 256, 1, 1, 0);
    initBN(conv4_var, conv4_mean, conv4_beta, 32);
    long conv5 = initConvAndFc(conv5_wb, 32, 32, 3, 1, 1);
    initBN(conv5_var, conv5_mean, conv5_beta, 32);
    long conv6 = initConvAndFc(conv6_wb, 32, 32, 3, 1, 1);
    initBN(conv6_var, conv6_mean, conv6_beta, 32);

    long conv7 = initConvAndFc(conv7_wb, 256, 96, 1, 1, 0);

    long conv8 = initConvAndFc(conv8_wb, 256, 0, 0, 0, 0);

    long dataNumber[28] = {conv1, 32, 32, 32, conv2, 32, 32, 32, conv3, 32, 32, 32, conv4, 32, 32, 32,
                           conv5, 32, 32, 32, conv6, 32, 32, 32, conv7, 256, conv8, 0};

    mydataFmt *pointTeam[28] = {conv1_wb->pdata, conv1_var->pdata, conv1_mean->pdata, conv1_beta->pdata, \
                            conv2_wb->pdata, conv2_var->pdata, conv2_mean->pdata, conv2_beta->pdata, \
                            conv3_wb->pdata, conv3_var->pdata, conv3_mean->pdata, conv3_beta->pdata, \
                            conv4_wb->pdata, conv4_var->pdata, conv4_mean->pdata, conv4_beta->pdata, \
                            conv5_wb->pdata, conv5_var->pdata, conv5_mean->pdata, conv5_beta->pdata, \
                            conv6_wb->pdata, conv6_var->pdata, conv6_mean->pdata, conv6_beta->pdata, \
                            conv7_wb->pdata, conv7_wb->pbias, \
                            conv8_wb->pdata, conv8_wb->pbias};

    readData(filepath, dataNumber, pointTeam, 28);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 35 x 35 x 32
    convolution(conv1_wb, input, conv1_out);
    BatchNorm(conv1_out, conv1_var, conv1_mean, conv1_beta);
    relu(conv1_out, conv1_wb->pbias);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 35 x 35 x 32
    convolution(conv2_wb, input, conv2_out);
    BatchNorm(conv2_out, conv2_var, conv2_mean, conv2_beta);
    relu(conv2_out, conv2_wb->pbias);
    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 35 x 35 x 32
    convolution(conv3_wb, conv2_out, conv3_out);
    BatchNorm(conv3_out, conv3_var, conv3_mean, conv3_beta);
    relu(conv3_out, conv3_wb->pbias);

    convolutionInit(conv4_wb, input, conv4_out);
    //conv4 35 x 35 x 32
    convolution(conv4_wb, input, conv4_out);
    BatchNorm(conv4_out, conv4_var, conv4_mean, conv4_beta);
    relu(conv4_out, conv4_wb->pbias);
    convolutionInit(conv5_wb, conv4_out, conv5_out);
    //conv5 35 x 35 x 32
    convolution(conv5_wb, conv4_out, conv5_out);
    BatchNorm(conv5_out, conv5_var, conv5_mean, conv5_beta);
    relu(conv5_out, conv5_wb->pbias);
    convolutionInit(conv6_wb, conv5_out, conv6_out);
    //conv6 35 x 35 x 32
    convolution(conv6_wb, conv5_out, conv6_out);
    BatchNorm(conv6_out, conv6_var, conv6_mean, conv6_beta);
    relu(conv6_out, conv6_wb->pbias);

    conv_mergeInit(conv7_out, conv1_out, conv3_out, conv6_out);
    //35 × 35 × 96
    conv_merge(conv7_out, conv1_out, conv3_out, conv6_out);

    convolutionInit(conv7_wb, conv7_out, conv8_out);
    //35*35*256
    convolution(conv7_wb, conv7_out, conv8_out);
    addbias(conv8_out, conv7_wb->pbias);

    mulandaddInit(input, conv8_out, output, scale);
    mulandadd(input, conv8_out, output, scale);
    relu(output, conv8_wb->pbias);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);
    freepBox(conv7_out);
    freepBox(conv8_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);
    freeWeight(conv7_wb);
    freeWeight(conv8_wb);

    freeBN(conv1_var);
    freeBN(conv1_mean);
    freeBN(conv1_beta);

    freeBN(conv2_var);
    freeBN(conv2_mean);
    freeBN(conv2_beta);

    freeBN(conv3_var);
    freeBN(conv3_mean);
    freeBN(conv3_beta);

    freeBN(conv4_var);
    freeBN(conv4_mean);
    freeBN(conv4_beta);

    freeBN(conv5_var);
    freeBN(conv5_mean);
    freeBN(conv5_beta);

    freeBN(conv6_var);
    freeBN(conv6_mean);
    freeBN(conv6_beta);
}

void facenet::Reduction_A(pBox *input, pBox *output) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;

    struct pBox *pooling1_out = new pBox;

    struct BN *conv1_var = new BN;
    struct BN *conv1_mean = new BN;
    struct BN *conv1_beta = new BN;
    struct BN *conv2_var = new BN;
    struct BN *conv2_mean = new BN;
    struct BN *conv2_beta = new BN;
    struct BN *conv3_var = new BN;
    struct BN *conv3_mean = new BN;
    struct BN *conv3_beta = new BN;
    struct BN *conv4_var = new BN;
    struct BN *conv4_mean = new BN;
    struct BN *conv4_beta = new BN;


    long conv1 = initConvAndFc(conv1_wb, 384, 256, 3, 2, 0);
    initBN(conv1_var, conv1_mean, conv1_beta, 384);

    long conv2 = initConvAndFc(conv2_wb, 192, 256, 1, 1, 0);
    initBN(conv2_var, conv2_mean, conv2_beta, 192);
    long conv3 = initConvAndFc(conv3_wb, 192, 192, 3, 1, 0);
    initBN(conv3_var, conv3_mean, conv3_beta, 192);
    long conv4 = initConvAndFc(conv4_wb, 256, 192, 3, 2, 0);
    initBN(conv4_var, conv4_mean, conv4_beta, 256);
    long dataNumber[16] = {conv1, 384, 384, 384, conv2, 192, 192, 192, conv3, 192, 192, 192, conv4, 256, 256, 256};

    mydataFmt *pointTeam[16] = {conv1_wb->pdata, conv1_var->pdata, conv1_mean->pdata, conv1_beta->pdata, \
                            conv2_wb->pdata, conv2_var->pdata, conv2_mean->pdata, conv2_beta->pdata, \
                            conv3_wb->pdata, conv3_var->pdata, conv3_mean->pdata, conv3_beta->pdata, \
                            conv4_wb->pdata, conv4_var->pdata, conv4_mean->pdata, conv4_beta->pdata};
    string filename = "../model_" + to_string(Num) + "/Mixed_6a_list.txt";
    readData(filename, dataNumber, pointTeam, 16);

    maxPoolingInit(input, pooling1_out, 3, 2);
    // 17*17*256
    maxPooling(input, pooling1_out, 3, 2);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 17 x 17 x 384
    convolution(conv1_wb, input, conv1_out);
    BatchNorm(conv1_out, conv1_var, conv1_mean, conv1_beta);
    relu(conv1_out, conv1_wb->pbias);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 35 x 35 x 192
    convolution(conv2_wb, input, conv2_out);
    BatchNorm(conv2_out, conv2_var, conv2_mean, conv2_beta);
    relu(conv2_out, conv2_wb->pbias);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 35 x 35 x 192
    convolution(conv3_wb, conv2_out, conv3_out);
    BatchNorm(conv3_out, conv3_var, conv3_mean, conv3_beta);
    relu(conv3_out, conv3_wb->pbias);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 17 x 17 x 256
    convolution(conv4_wb, conv3_out, conv4_out);
    BatchNorm(conv4_out, conv4_var, conv4_mean, conv4_beta);
    relu(conv4_out, conv4_wb->pbias);
    conv_mergeInit(output, pooling1_out, conv1_out, conv4_out);
    //17×17×896
    conv_merge(output, pooling1_out, conv1_out, conv4_out);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);

    freepBox(pooling1_out);

    freeBN(conv1_var);
    freeBN(conv1_mean);
    freeBN(conv1_beta);
    freeBN(conv2_var);
    freeBN(conv2_mean);
    freeBN(conv2_beta);
    freeBN(conv3_var);
    freeBN(conv3_mean);
    freeBN(conv3_beta);
    freeBN(conv4_var);
    freeBN(conv4_mean);
    freeBN(conv4_beta);
}

void facenet::Inception_resnet_B(pBox *input, pBox *output, string filepath, float scale) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;

    struct BN *conv1_var = new BN;
    struct BN *conv1_mean = new BN;
    struct BN *conv1_beta = new BN;
    struct BN *conv2_var = new BN;
    struct BN *conv2_mean = new BN;
    struct BN *conv2_beta = new BN;
    struct BN *conv3_var = new BN;
    struct BN *conv3_mean = new BN;
    struct BN *conv3_beta = new BN;
    struct BN *conv4_var = new BN;
    struct BN *conv4_mean = new BN;
    struct BN *conv4_beta = new BN;


    long conv1 = initConvAndFc(conv1_wb, 128, 896, 1, 1, 0);
    initBN(conv1_var, conv1_mean, conv1_beta, 128);

    long conv2 = initConvAndFc(conv2_wb, 128, 896, 1, 1, 0);
    initBN(conv2_var, conv2_mean, conv2_beta, 128);
    long conv3 = initConvAndFc(conv3_wb, 128, 128, 0, 1, -1, 7, 1, 3, 0);//[1,7]
    initBN(conv3_var, conv3_mean, conv3_beta, 128);
    long conv4 = initConvAndFc(conv4_wb, 128, 128, 0, 1, -1, 1, 7, 0, 3);//[7,1]
    initBN(conv4_var, conv4_mean, conv4_beta, 128);

    long conv5 = initConvAndFc(conv5_wb, 896, 256, 1, 1, 0);

    long conv6 = initConvAndFc(conv6_wb, 896, 0, 0, 0, 0);

    long dataNumber[20] = {conv1, 128, 128, 128, conv2, 128, 128, 128, conv3, 128, 128, 128, conv4, 128, 128, 128,
                           conv5, 896, conv6, 0};

    mydataFmt *pointTeam[20] = {conv1_wb->pdata, conv1_var->pdata, conv1_mean->pdata, conv1_beta->pdata, \
                            conv2_wb->pdata, conv2_var->pdata, conv2_mean->pdata, conv2_beta->pdata, \
                            conv3_wb->pdata, conv3_var->pdata, conv3_mean->pdata, conv3_beta->pdata, \
                            conv4_wb->pdata, conv4_var->pdata, conv4_mean->pdata, conv4_beta->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias, \
                            conv6_wb->pdata, conv6_wb->pbias};


    readData(filepath, dataNumber, pointTeam, 20);


    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 17*17*128
    convolution(conv1_wb, input, conv1_out);
    BatchNorm(conv1_out, conv1_var, conv1_mean, conv1_beta);
    relu(conv1_out, conv1_wb->pbias);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 17*17*128
    convolution(conv2_wb, input, conv2_out);
    BatchNorm(conv2_out, conv2_var, conv2_mean, conv2_beta);
    relu(conv2_out, conv2_wb->pbias);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 17*17*128
    convolution(conv3_wb, conv2_out, conv3_out);
    BatchNorm(conv3_out, conv3_var, conv3_mean, conv3_beta);
    relu(conv3_out, conv3_wb->pbias);
    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 17*17*128
    convolution(conv4_wb, conv3_out, conv4_out);
    BatchNorm(conv4_out, conv4_var, conv4_mean, conv4_beta);
    relu(conv4_out, conv4_wb->pbias);

    conv_mergeInit(conv5_out, conv1_out, conv4_out);
    //17*17*256
    conv_merge(conv5_out, conv1_out, conv4_out);

    convolutionInit(conv5_wb, conv5_out, conv6_out);
    //conv5 17*17*896
    convolution(conv5_wb, conv5_out, conv6_out);
    addbias(conv6_out, conv5_wb->pbias);

    mulandaddInit(input, conv6_out, output, scale);
    mulandadd(input, conv6_out, output, scale);
    relu(output, conv6_wb->pbias);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);

    freeBN(conv1_var);
    freeBN(conv1_mean);
    freeBN(conv1_beta);
    freeBN(conv2_var);
    freeBN(conv2_mean);
    freeBN(conv2_beta);
    freeBN(conv3_var);
    freeBN(conv3_mean);
    freeBN(conv3_beta);
    freeBN(conv4_var);
    freeBN(conv4_mean);
    freeBN(conv4_beta);
}

void facenet::Reduction_B(pBox *input, pBox *output) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;
    pBox *conv7_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;
    struct Weight *conv7_wb = new Weight;

    struct pBox *pooling1_out = new pBox;

    struct BN *conv1_var = new BN;
    struct BN *conv1_mean = new BN;
    struct BN *conv1_beta = new BN;
    struct BN *conv2_var = new BN;
    struct BN *conv2_mean = new BN;
    struct BN *conv2_beta = new BN;
    struct BN *conv3_var = new BN;
    struct BN *conv3_mean = new BN;
    struct BN *conv3_beta = new BN;
    struct BN *conv4_var = new BN;
    struct BN *conv4_mean = new BN;
    struct BN *conv4_beta = new BN;
    struct BN *conv5_var = new BN;
    struct BN *conv5_mean = new BN;
    struct BN *conv5_beta = new BN;
    struct BN *conv6_var = new BN;
    struct BN *conv6_mean = new BN;
    struct BN *conv6_beta = new BN;
    struct BN *conv7_var = new BN;
    struct BN *conv7_mean = new BN;
    struct BN *conv7_beta = new BN;


    long conv1 = initConvAndFc(conv1_wb, 256, 896, 1, 1, 0);
    initBN(conv1_var, conv1_mean, conv1_beta, 256);
    long conv2 = initConvAndFc(conv2_wb, 384, 256, 3, 2, 0);
    initBN(conv2_var, conv2_mean, conv2_beta, 384);

    long conv3 = initConvAndFc(conv3_wb, 256, 896, 1, 1, 0);
    initBN(conv3_var, conv3_mean, conv3_beta, 256);
    long conv4 = initConvAndFc(conv4_wb, 256, 256, 3, 2, 0);
    initBN(conv4_var, conv4_mean, conv4_beta, 256);

    long conv5 = initConvAndFc(conv5_wb, 256, 896, 1, 1, 0);
    initBN(conv5_var, conv5_mean, conv5_beta, 256);
    long conv6 = initConvAndFc(conv6_wb, 256, 256, 3, 1, 1);
    initBN(conv6_var, conv6_mean, conv6_beta, 256);
    long conv7 = initConvAndFc(conv7_wb, 256, 256, 3, 2, 0);
    initBN(conv7_var, conv7_mean, conv7_beta, 256);

    long dataNumber[28] = {conv1, 256, 256, 256, conv2, 384, 384, 384, conv3, 256, 256, 256, conv4, 256, 256, 256,
                           conv5, 256, 256, 256, conv6, 256, 256, 256, conv7, 256, 256, 256};

    mydataFmt *pointTeam[28] = {conv1_wb->pdata, conv1_var->pdata, conv1_mean->pdata, conv1_beta->pdata, \
                            conv2_wb->pdata, conv2_var->pdata, conv2_mean->pdata, conv2_beta->pdata, \
                            conv3_wb->pdata, conv3_var->pdata, conv3_mean->pdata, conv3_beta->pdata, \
                            conv4_wb->pdata, conv4_var->pdata, conv4_mean->pdata, conv4_beta->pdata, \
                            conv5_wb->pdata, conv5_var->pdata, conv5_mean->pdata, conv5_beta->pdata, \
                            conv6_wb->pdata, conv6_var->pdata, conv6_mean->pdata, conv6_beta->pdata, \
                            conv7_wb->pdata, conv7_var->pdata, conv7_mean->pdata, conv7_beta->pdata};
    string filename = "../model_" + to_string(Num) + "/Mixed_7a_list.txt";
    readData(filename, dataNumber, pointTeam, 28);


    maxPoolingInit(input, pooling1_out, 3, 2, 1);
    // 8*8*896
    maxPooling(input, pooling1_out, 3, 2);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 17 x 17 x 256
    convolution(conv1_wb, input, conv1_out);
    BatchNorm(conv1_out, conv1_var, conv1_mean, conv1_beta);
    relu(conv1_out, conv1_wb->pbias);

    convolutionInit(conv2_wb, conv1_out, conv2_out);
    //conv2 8 x 8 x 384
    convolution(conv2_wb, conv1_out, conv2_out);
    BatchNorm(conv2_out, conv2_var, conv2_mean, conv2_beta);
    relu(conv2_out, conv2_wb->pbias);

    convolutionInit(conv3_wb, input, conv3_out);
    //conv3 17 x 17 x 256
    convolution(conv3_wb, input, conv3_out);
    BatchNorm(conv3_out, conv3_var, conv3_mean, conv3_beta);
    relu(conv3_out, conv3_wb->pbias);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 8 x 8 x 256
    convolution(conv4_wb, conv3_out, conv4_out);
    BatchNorm(conv4_out, conv4_var, conv4_mean, conv4_beta);
    relu(conv4_out, conv4_wb->pbias);

    convolutionInit(conv5_wb, input, conv5_out);
    //conv5 17 x 17 x 256
    convolution(conv5_wb, input, conv5_out);
    BatchNorm(conv5_out, conv5_var, conv5_mean, conv5_beta);
    relu(conv5_out, conv5_wb->pbias);

    convolutionInit(conv6_wb, conv5_out, conv6_out);
    //conv6 17 x 17 x 256
    convolution(conv6_wb, conv5_out, conv6_out);
    BatchNorm(conv6_out, conv6_var, conv6_mean, conv6_beta);
    relu(conv6_out, conv6_wb->pbias);

    convolutionInit(conv7_wb, conv6_out, conv7_out);
    //conv6 8 x 8 x 256
    convolution(conv7_wb, conv6_out, conv7_out);
    BatchNorm(conv7_out, conv7_var, conv7_mean, conv7_beta);
    relu(conv7_out, conv7_wb->pbias);

    conv_mergeInit(output, conv2_out, conv4_out, conv7_out, pooling1_out);
    //8*8*1792
    conv_merge(output, conv2_out, conv4_out, conv7_out, pooling1_out);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);
    freepBox(conv7_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);
    freeWeight(conv7_wb);

    freepBox(pooling1_out);

    freeBN(conv1_var);
    freeBN(conv1_mean);
    freeBN(conv1_beta);
    freeBN(conv2_var);
    freeBN(conv2_mean);
    freeBN(conv2_beta);
    freeBN(conv3_var);
    freeBN(conv3_mean);
    freeBN(conv3_beta);
    freeBN(conv4_var);
    freeBN(conv4_mean);
    freeBN(conv4_beta);
    freeBN(conv5_var);
    freeBN(conv5_mean);
    freeBN(conv5_beta);
    freeBN(conv6_var);
    freeBN(conv6_mean);
    freeBN(conv6_beta);
    freeBN(conv7_var);
    freeBN(conv7_mean);
    freeBN(conv7_beta);
}

void facenet::Inception_resnet_C(pBox *input, pBox *output, string filepath, float scale) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;

    struct BN *conv1_var = new BN;
    struct BN *conv1_mean = new BN;
    struct BN *conv1_beta = new BN;
    struct BN *conv2_var = new BN;
    struct BN *conv2_mean = new BN;
    struct BN *conv2_beta = new BN;
    struct BN *conv3_var = new BN;
    struct BN *conv3_mean = new BN;
    struct BN *conv3_beta = new BN;
    struct BN *conv4_var = new BN;
    struct BN *conv4_mean = new BN;
    struct BN *conv4_beta = new BN;


    long conv1 = initConvAndFc(conv1_wb, 192, 1792, 1, 1, 0);
    initBN(conv1_var, conv1_mean, conv1_beta, 192);
    long conv2 = initConvAndFc(conv2_wb, 192, 1792, 1, 1, 0);
    initBN(conv2_var, conv2_mean, conv2_beta, 192);
    long conv3 = initConvAndFc(conv3_wb, 192, 192, 0, 1, -1, 3, 1, 1, 0);
    initBN(conv3_var, conv3_mean, conv3_beta, 192);
    long conv4 = initConvAndFc(conv4_wb, 192, 192, 0, 1, -1, 1, 3, 0, 1);
    initBN(conv4_var, conv4_mean, conv4_beta, 192);

    long conv5 = initConvAndFc(conv5_wb, 1792, 384, 1, 1, 0);

    long conv6 = initConvAndFc(conv6_wb, 1792, 0, 0, 0, 0);

    long dataNumber[20] = {conv1, 192, 192, 192, conv2, 192, 192, 192, conv3, 192, 192, 192, conv4, 192, 192, 192,
                           conv5, 1792, conv6, 0};


    mydataFmt *pointTeam[20] = {conv1_wb->pdata, conv1_var->pdata, conv1_mean->pdata, conv1_beta->pdata, \
                            conv2_wb->pdata, conv2_var->pdata, conv2_mean->pdata, conv2_beta->pdata, \
                            conv3_wb->pdata, conv3_var->pdata, conv3_mean->pdata, conv3_beta->pdata, \
                            conv4_wb->pdata, conv4_var->pdata, conv4_mean->pdata, conv4_beta->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias, \
                            conv6_wb->pdata, conv6_wb->pbias};

//    string filename = "../model_128/Repeat_2_list.txt";
//    int length = sizeof(dataNumber) / sizeof(*dataNumber);
    readData(filepath, dataNumber, pointTeam, 20);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 8 x 8 x 192
    convolution(conv1_wb, input, conv1_out);
    BatchNorm(conv1_out, conv1_var, conv1_mean, conv1_beta);
    relu(conv1_out, conv1_wb->pbias);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 8 x 8 x 192
    convolution(conv2_wb, input, conv2_out);
    BatchNorm(conv2_out, conv2_var, conv2_mean, conv2_beta);
    relu(conv2_out, conv2_wb->pbias);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 8 x 8 x 192
    convolution(conv3_wb, conv2_out, conv3_out);
    BatchNorm(conv3_out, conv3_var, conv3_mean, conv3_beta);
    relu(conv3_out, conv3_wb->pbias);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 8 x 8 x 192
    convolution(conv4_wb, conv3_out, conv4_out);
    BatchNorm(conv4_out, conv4_var, conv4_mean, conv4_beta);
    relu(conv4_out, conv4_wb->pbias);

    conv_mergeInit(conv5_out, conv1_out, conv4_out);
    // 8*8*384
    conv_merge(conv5_out, conv1_out, conv4_out);

    convolutionInit(conv5_wb, conv5_out, conv6_out);
    //conv5 8 x 8 x 1792
    convolution(conv5_wb, conv5_out, conv6_out);
    addbias(conv6_out, conv5_wb->pbias);

    mulandaddInit(input, conv6_out, output, scale);
    mulandadd(input, conv6_out, output, scale);
    relu(output, conv6_wb->pbias);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);

    freeBN(conv1_var);
    freeBN(conv1_mean);
    freeBN(conv1_beta);
    freeBN(conv2_var);
    freeBN(conv2_mean);
    freeBN(conv2_beta);
    freeBN(conv3_var);
    freeBN(conv3_mean);
    freeBN(conv3_beta);
    freeBN(conv4_var);
    freeBN(conv4_mean);
    freeBN(conv4_beta);
}

void facenet::Inception_resnet_C_None(pBox *input, pBox *output, string filepath) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;

    struct BN *conv1_var = new BN;
    struct BN *conv1_mean = new BN;
    struct BN *conv1_beta = new BN;
    struct BN *conv2_var = new BN;
    struct BN *conv2_mean = new BN;
    struct BN *conv2_beta = new BN;
    struct BN *conv3_var = new BN;
    struct BN *conv3_mean = new BN;
    struct BN *conv3_beta = new BN;
    struct BN *conv4_var = new BN;
    struct BN *conv4_mean = new BN;
    struct BN *conv4_beta = new BN;

    long conv1 = initConvAndFc(conv1_wb, 192, 1792, 1, 1, 0);
    initBN(conv1_var, conv1_mean, conv1_beta, 192);
    long conv2 = initConvAndFc(conv2_wb, 192, 1792, 1, 1, 0);
    initBN(conv2_var, conv2_mean, conv2_beta, 192);
    long conv3 = initConvAndFc(conv3_wb, 192, 192, 0, 1, -1, 3, 1, 1, 0);
    initBN(conv3_var, conv3_mean, conv3_beta, 192);
    long conv4 = initConvAndFc(conv4_wb, 192, 192, 0, 1, -1, 1, 3, 0, 1);
    initBN(conv4_var, conv4_mean, conv4_beta, 192);
    long conv5 = initConvAndFc(conv5_wb, 1792, 384, 1, 1, 0);

    long dataNumber[18] = {conv1, 192, 192, 192, conv2, 192, 192, 192, conv3, 192, 192, 192, conv4, 192, 192, 192,
                           conv5, 1792};


    mydataFmt *pointTeam[18] = {conv1_wb->pdata, conv1_var->pdata, conv1_mean->pdata, conv1_beta->pdata, \
                            conv2_wb->pdata, conv2_var->pdata, conv2_mean->pdata, conv2_beta->pdata, \
                            conv3_wb->pdata, conv3_var->pdata, conv3_mean->pdata, conv3_beta->pdata, \
                            conv4_wb->pdata, conv4_var->pdata, conv4_mean->pdata, conv4_beta->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias};

//    string filename = "../model_128/Repeat_2_list.txt";
//    int length = sizeof(dataNumber) / sizeof(*dataNumber);
    readData(filepath, dataNumber, pointTeam, 18);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 8 x 8 x 192
    convolution(conv1_wb, input, conv1_out);
    BatchNorm(conv1_out, conv1_var, conv1_mean, conv1_beta);
    relu(conv1_out, conv1_wb->pbias);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 8 x 8 x 192
    convolution(conv2_wb, input, conv2_out);
    BatchNorm(conv2_out, conv2_var, conv2_mean, conv2_beta);
    relu(conv2_out, conv2_wb->pbias);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 8 x 8 x 192
    convolution(conv3_wb, conv2_out, conv3_out);
    BatchNorm(conv3_out, conv3_var, conv3_mean, conv3_beta);
    relu(conv3_out, conv3_wb->pbias);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 8 x 8 x 192
    convolution(conv4_wb, conv3_out, conv4_out);
    BatchNorm(conv4_out, conv4_var, conv4_mean, conv4_beta);
    relu(conv4_out, conv4_wb->pbias);

    conv_mergeInit(conv5_out, conv1_out, conv4_out);
    // 8*8*384
    conv_merge(conv5_out, conv1_out, conv4_out);

    convolutionInit(conv5_wb, conv5_out, conv6_out);
    //conv5 8 x 8 x 1792
    convolution(conv5_wb, conv5_out, conv6_out);
    addbias(conv6_out, conv5_wb->pbias);

    mulandaddInit(input, conv6_out, output, 1);
    mulandadd(input, conv6_out, output);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);

    freeBN(conv1_var);
    freeBN(conv1_mean);
    freeBN(conv1_beta);
    freeBN(conv2_var);
    freeBN(conv2_mean);
    freeBN(conv2_beta);
    freeBN(conv3_var);
    freeBN(conv3_mean);
    freeBN(conv3_beta);
    freeBN(conv4_var);
    freeBN(conv4_mean);
    freeBN(conv4_beta);
}

void facenet::AveragePooling(pBox *input, pBox *output) {
//    cout << "size:" << input->height << endl;
    avePoolingInit(input, output, input->height, 2);
    avePooling(input, output, input->height, 2);
}

void facenet::Flatten(pBox *input, pBox *output) {
    output->width = input->channel;
    output->height = 1;
    output->channel = 1;
    output->pdata = (mydataFmt *) malloc(output->channel * output->width * output->height * sizeof(mydataFmt));
    if (output->pdata == NULL)cout << "the maxPoolingInit is failed!!" << endl;
    memcpy(output->pdata, input->pdata, output->channel * output->width * output->height * sizeof(mydataFmt));
}

//参数还未设置
void facenet::fully_connect(pBox *input, pBox *output, string filepath) {
    struct Weight *conv1_wb = new Weight;
    struct BN *conv1_var = new BN;
    struct BN *conv1_mean = new BN;
    struct BN *conv1_beta = new BN;
    long conv1 = initConvAndFc(conv1_wb, Num, 1792, input->height, 1, 0);
    initBN(conv1_var, conv1_mean, conv1_beta, Num);
    long dataNumber[4] = {conv1, Num, Num, Num};

//    cout << to_string(sum) << endl;
    mydataFmt *pointTeam[4] = {conv1_wb->pdata, conv1_var->pdata, conv1_mean->pdata, conv1_beta->pdata};
//    string filename = "../model_128/Bottleneck_list.txt";
//    int length = sizeof(dataNumber) / sizeof(*dataNumber);
    readData(filepath, dataNumber, pointTeam, 4);

    fullconnectInit(conv1_wb, output);

    //conv1 8 x 8 x 192
    fullconnect(conv1_wb, input, output);
    BatchNorm(output, conv1_var, conv1_mean, conv1_beta);

//    relu(output, conv1_wb->pbias, prelu_gmma1->pdata);

    freeWeight(conv1_wb);
    freeBN(conv1_var);
    freeBN(conv1_mean);
    freeBN(conv1_beta);
}

void facenet::conv_mergeInit(pBox *output, pBox *c1, pBox *c2, pBox *c3, pBox *c4) {
    output->channel = 0;
    output->height = c1->height;
    output->width = c1->width;
    if (c1 != 0) {
        output->channel = c1->channel;
        if (c2 != 0) {
            output->channel += c2->channel;
            if (c3 != 0) {
                output->channel += c3->channel;
                if (c4 != 0) {
                    output->channel += c4->channel;
                }
            }
        }
    } else { cout << "conv_mergeInit" << endl; }
    output->pdata = (mydataFmt *) malloc(output->width * output->height * output->channel * sizeof(mydataFmt));
    if (output->pdata == NULL)cout << "the conv_mergeInit is failed!!" << endl;
    memset(output->pdata, 0, output->width * output->height * output->channel * sizeof(mydataFmt));
}

void facenet::conv_merge(pBox *output, pBox *c1, pBox *c2, pBox *c3, pBox *c4) {
//    cout << "output->channel:" << output->channel << endl;
    if (c1 != 0) {
        long count1 = c1->height * c1->width * c1->channel;
        //output->pdata = c1->pdata;
        for (long i = 0; i < count1; i++) {
            output->pdata[i] = c1->pdata[i];
        }
        if (c2 != 0) {
            long count2 = c2->height * c2->width * c2->channel;
            for (long i = 0; i < count2; i++) {
                output->pdata[count1 + i] = c2->pdata[i];
            }
            if (c3 != 0) {
                long count3 = c3->height * c3->width * c3->channel;
                for (long i = 0; i < count3; i++) {
                    output->pdata[count1 + count2 + i] = c3->pdata[i];
                }
                if (c4 != 0) {
                    long count4 = c4->height * c4->width * c4->channel;
                    for (long i = 0; i < count4; i++) {
                        output->pdata[count1 + count2 + count3 + i] = c4->pdata[i];
                    }
                }
            }
        }
    } else { cout << "conv_mergeInit" << endl; }
//    cout << "output->pdata:" << *(output->pdata) << endl;
}

void facenet::mulandaddInit(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale) {
    outpBox->channel = temppbox->channel;
    outpBox->width = temppbox->width;
    outpBox->height = temppbox->height;
    outpBox->pdata = (mydataFmt *) malloc(outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
    if (outpBox->pdata == NULL)cout << "the mulandaddInit is failed!!" << endl;
    memset(outpBox->pdata, 0, outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
}

void facenet::mulandadd(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale) {
    mydataFmt *ip = inpbox->pdata;
    mydataFmt *tp = temppbox->pdata;
    mydataFmt *op = outpBox->pdata;
    long dis = inpbox->width * inpbox->height * inpbox->channel;
    for (long i = 0; i < dis; i++) {
        op[i] = ip[i] + tp[i] * scale;
    }
}

void facenet::prewhiten(Mat &image) {
    double mean, stddev, sqr, stddev_adj;
    int size;

    Mat temp_m, temp_sd;
    meanStdDev(image, temp_m, temp_sd);
    mean = temp_m.at<double>(0, 0);
    stddev = temp_sd.at<double>(0, 0);
    size = image.cols * image.rows * image.channels();
    sqr = sqrt(double(size));
    if (stddev > 1.0 / sqr) {
        stddev_adj = stddev;
    } else {
        stddev_adj = 1.0 / sqr;
    }
    Mat temp_image(image.rows, image.cols, CV_64F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            image.at<uchar>(i, j);
            temp_image.at<Vec3b>(i, j)[0] = (image.at<Vec3b>(i, j)[0] - mean) / stddev_adj;
            temp_image.at<Vec3b>(i, j)[0] = (image.at<Vec3b>(i, j)[0] - mean) / stddev_adj;
            temp_image.at<Vec3b>(i, j)[0] = (image.at<Vec3b>(i, j)[0] - mean) / stddev_adj;
            cout << 1 << endl;
        }
    }
//    double max, min;
//    minMaxLoc(temp_image, &min, &max);
//    for (int i = 0; i < image.rows; i++) {
//        for (int j = 0; j < image.cols; j++) {
//            double pixelVal = temp_image.at<double>(i, j);
//            image.at<uchar>(i, j) = temp_image.at<double>(i, j);
//        }
//    }
//    imshow("New Image", image);
//    waitKey(0);
}
