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
    outFile.open("../emb_csv/" + to_string(num) + ".csv", ios::out); // 打开模式可省略
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

void run_mtcnn(Mat &image, vector<Rect> &vecRect, vector<Point> &vecPoint) {
//    vector<Point> vecPoint;
    mtcnn find(image.rows, image.cols);
    find.findFace(image, vecRect, vecPoint);
//    for (int i = 0; i < vecRect.size(); ++i) {
//        rectangle(image, vecPoint[7 * i + 0], vecPoint[7 * i + 1], Scalar(0, 0, 255), 2, 8, 0);
//        for (int num = 0; num < 5; num++)
//            circle(image, vecPoint[7 * i + num + 2], 2, Scalar(0, 255, 255),
//                   -1);
//    }
}

void run_facenet(Mat &image, vector<Rect> &vecRect, int csv_num = 0) {
    for (int i = 0; i < vecRect.size(); ++i) {
        Mat fourthImage;
        resize(image(vecRect[i]), fourthImage, Size(160, 160), 0, 0, cv::INTER_LINEAR);
        facenet ggg;
//        mydataFmt *o = new mydataFmt[Num];
        vector<mydataFmt> n;
        vector<vector<mydataFmt>> o;
        ggg.run(fourthImage, n, i);

//        write_emb_csv(n, i);
//        return;
        if (csv_num != 0) {
            load_emb_csv(csv_num, o);
            for (int j = 0; j < o.size(); ++j) {
                float result = compare(n, o[j]);
                cout << "-------------------" << endl;
                cout << result << endl;
                if (result < 0.45)
                    cout << "it's me" << endl;
                else
                    cout << "unknow" << endl;
                cout << j << endl;
            }
        } else {
            imwrite("../emb_img/" + to_string(i) + ".jpg", fourthImage);
            write_emb_csv(n, i);
        }
    }
}

float test_compare(vector<mydataFmt> &lineArray0, vector<mydataFmt> &lineArray1) {
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

void test_facenet(Mat &image, vector<Rect> &vecRect, vector<mydataFmt> &n) {
    Mat fourthImage;
    resize(image(vecRect[0]), fourthImage, Size(160, 160), 0, 0, cv::INTER_LINEAR);
    facenet ggg;
//        mydataFmt *o = new mydataFmt[Num];
//    vector<mydataFmt> n;
    vector<vector<mydataFmt>> o;
    ggg.run(fourthImage, n, 0);
}

void run() {
    int b = 0;
    if (b == 0) {

//                Mat image = imread("../40.jpg");
        Mat image = imread("../3.jpeg");
//        Mat image = imread("../4.jpeg");
//        Mat image = imread("../xiena.jpeg");
//        Mat image = imread("../hejiong.jpeg");
//        Mat image = imread("../libingbing.jpeg");
//        Mat image = imread("../zhangjie.jpg");
//        Mat image = imread("../Kong_Weiye.jpg");
//        Mat image = imread("../kkk.jpg");
//        Mat image = imread("../20.png");
//        Mat image = imread("../emb_img/0.jpg");

        clock_t start;
        start = clock();
        vector<Rect> vecRect;
        vector<Point> vecPoint;
        run_mtcnn(image, vecRect, vecPoint);

        run_facenet(image, vecRect, 13);// 第三个参数csv数量，如果为0，则是保存emb到csv功能

        imshow("result", image);
        imwrite("../result.jpg", image);
        start = clock() - start;
        //    cout<<"time is  "<<start/10e3<<endl;
        cout << "time is " << (double) start / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        waitKey(5000);
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

void test() {
    Mat image0 = imread("../kkk.jpg");
    Mat image1 = imread("../hejiong0.jpeg");

    clock_t start;
    start = clock();
    vector<Rect> vecRect0, vecRect1;
    vector<mydataFmt> n0, n1;
    vector<Point> vecPoint0, vecPoint1;
    run_mtcnn(image0, vecRect0, vecPoint0);
    test_facenet(image0, vecRect0, n0);
    run_mtcnn(image1, vecRect1, vecPoint1);
    test_facenet(image1, vecRect1, n1);

    for (int i = 0; i < vecRect0.size(); ++i) {
        rectangle(image0, vecPoint0[7 * i + 0], vecPoint0[7 * i + 1], Scalar(0, 0, 255), 2, 8, 0);
        for (int num = 0; num < 5; num++)
            circle(image0, vecPoint0[7 * i + num + 2], 2, Scalar(0, 255, 255),
                   -1);
    }
    for (int i = 0; i < vecRect1.size(); ++i) {
        rectangle(image1, vecPoint1[7 * i + 0], vecPoint1[7 * i + 1], Scalar(0, 0, 255), 2, 8, 0);
        for (int num = 0; num < 5; num++)
            circle(image1, vecPoint1[7 * i + num + 2], 2, Scalar(0, 255, 255),
                   -1);
    }

    float result = test_compare(n0, n1);
    cout << "-------------------" << endl;
    cout << result << endl;
    if (result < 0.45)
        cout << "可能是同一个人" << endl;
    else
        cout << "很可能不是同一个人" << endl;

    imshow("result0", image0);
    imwrite("../result0.jpg", image0);
    imshow("result1", image1);
    imwrite("../result1.jpg", image1);
    start = clock() - start;
    //    cout<<"time is  "<<start/10e3<<endl;
    cout << "time is " << (double) start / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    waitKey(0);
    image0.release();
    image1.release();
}

int main() {
    for (int i = 0; i < 1; ++i) {
        test();
        cout << "==============================" << endl;
    }
}
