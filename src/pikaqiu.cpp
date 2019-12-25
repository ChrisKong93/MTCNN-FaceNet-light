#include "network.h"
#include "mtcnn.h"
#include "facenet.h"
#include <time.h>

void load_emb_csv(int num, vector<vector<mydataFmt>> &vecVec) {
    for (int i = 0; i < num; ++i) {
        ifstream inFile("../emb_csv/" + to_string(i) + ".csv", ios::in);
        string lineStr;
//                vector<vector<mydataFmt>> strArray;
        vector<mydataFmt> lineArray;
        while (getline(inFile, lineStr)) {
            // 打印整行字符串
//                    cout << lineStr << endl;
            // 存成二维表结构
            stringstream ss(lineStr);
            string str;
//                    vector<mydataFmt> lineArray;
            // 按照逗号分隔
//                    mydataFmt nnn = 0;
            while (getline(ss, str, ',')) {
                lineArray.push_back(atof(str.c_str()));
//                        cout << str << endl;
            }
//                    strArray.push_back(lineArray);
        }
        vecVec.push_back(lineArray);
    }
}

void write_emb_csv(vector<mydataFmt> &o, int num) {
    ofstream outFile;
    outFile.open("../emb_csv/" + to_string(num + 1) + ".csv", ios::out); // 打开模式可省略
    for (int l = 0; l < Num; ++l) {
//            cout << o[l] << endl;
        if (l == Num - 1) {
            outFile << o[l];
        } else {
            outFile << o[l] << ',';
        }
    }
    outFile << endl;
    outFile.close();
    cout << "write over!" << endl;
}

float compare(vector<mydataFmt> &lineArray0, vector<mydataFmt> &lineArray1) {
    mydataFmt sum = 0;
    for (int i = 0; i < Num; ++i) {
//        cout << lineArray0[i] << "===" << lineArray1[i] << endl;
        mydataFmt sub = lineArray0[i] - lineArray1[i];
        mydataFmt square = pow(sub, 2);
        sum += square;
    }
    mydataFmt result = sqrt(sum);
    return result;
}

void run_mtcnn(Mat &image, vector<Rect> &vecRect) {
    vector<Point> vecPoint;
    mtcnn find(image.rows, image.cols);
    find.findFace(image, vecRect, vecPoint);
    for (int i = 0; i < vecRect.size(); ++i) {
        rectangle(image, vecPoint[7 * i + 0], vecPoint[7 * i + 1], Scalar(0, 0, 255), 2, 8, 0);
        for (int num = 0; num < 5; num++)
            circle(image, vecPoint[7 * i + num + 2], 2, Scalar(0, 255, 255),
                   -1);
    }
}

void run_facenet(Mat &image, vector<Rect> &vecRect, int csv_num) {
    for (int i = 0; i < vecRect.size(); ++i) {
        Mat fourthImage;
        resize(image(vecRect[i]), fourthImage, Size(299, 299), 0, 0, cv::INTER_LINEAR);
        facenet ggg;
//        mydataFmt *o = new mydataFmt[Num];
        vector<mydataFmt> n;
        vector<vector<mydataFmt>> o;
        ggg.run(fourthImage, n, i);

//        write_emb_csv(n, i);
//        return;
        load_emb_csv(csv_num, o);
        for (int j = 0; j < o.size(); ++j) {
            float result = compare(n, o[j]);
            cout << result << endl;
            if (result < 0.85)
                cout << "it's me" << endl;
            else
                cout << "unknow" << endl;
        }
    }
}

void run() {
    int b = 0;
    if (b == 0) {
//        Mat image = imread("../40.jpg");
//        Mat image = imread("../3.jpeg");
//        Mat image = imread("../Kong_Weiye.jpg");
        Mat image = imread("../kkk.jpg");
//        Mat image = imread("../20.png");
//        Mat image = imread("../emb_img/0.jpg");

        clock_t start;
        start = clock();
        vector<Rect> vecRect;
        run_mtcnn(image, vecRect);
        run_facenet(image, vecRect, 13);

        imshow("result", image);
        imwrite("../result.jpg", image);
        start = clock() - start;
        //    cout<<"time is  "<<start/10e3<<endl;
        cout << "time is " << (double) start / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        waitKey(5000);
        image.release();
    } else if (b == 1) {
        Mat image = imread("../10.jpg");
//        Mat image = imread("../emb_img/0.jpg");
//        Mat image = imread("../20.png");
        Mat Image;
        resize(image, Image, Size(160, 160), 0, 0, cv::INTER_LINEAR);
        facenet ggg;
        mydataFmt *o = new mydataFmt[Num];
//        ggg.run(Image, o, 0);
//        imshow("result", Image);
        imwrite("../result.jpg", Image);

        for (int i = 0; i < Num; ++i) {
            cout << o[i] << endl;
        }

        waitKey(0);
        image.release();
    } else {
        Mat image;
        VideoCapture cap(0);
        if (!cap.isOpened())
            cout << "fail to open!" << endl;
        cap >> image;
        if (!image.data) {
            cout << "读取视频失败" << endl;
        }

        mtcnn find(image.rows, image.cols);
        clock_t start;
        int stop = 1200;
        //while (stop--) {
        while (true) {
            start = clock();
            cap >> image;
            vector<Rect> vecRect;
            vector<Point> vecPoint;
            find.findFace(image, vecRect, vecPoint);
            imshow("result", image);
            if (waitKey(1) >= 0) break;
            start = clock() - start;
            cout << "time is " << (double) start / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        }
        waitKey(0);
        image.release();
    }
}

int main() {
    for (int i = 0; i < 1; ++i) {
        run();
        cout << "==============================" << endl;
    }
}
