#include "network.h"
#include "mtcnn.h"
#include "facenet.h"
#include <time.h>

/**
 * 图片缩小
 * @param src 输入图片
 * @return 返回图片
 */
Mat RS(Mat &src) {
    int w = src.cols;
    int h = src.rows;
    int wtemp, htemp;
    Mat dst;
    cout << w << "\t" << h << endl;
    float threshold = 300.0;
    if (h > threshold) {
        wtemp = (int) (threshold / h * w);
        htemp = threshold;
        dst = Mat::zeros(htemp, wtemp, CV_8UC3); //我要转化为htemp*wtemp大小的
        resize(src, dst, dst.size());
    }
    cout << wtemp << "\t" << htemp << endl;
    cout << "-------------------" << endl;
    return dst;
}


/**
 * 对比两个人的emb值，计算空间欧氏距离
 * @param lineArray0 第一个人的emb值
 * @param lineArray1 第二个人的emb值
 * @return
 */
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

/**
 * 执行mtcnn网络
 * @param image 图片
 * @param vecRect 获取人脸框
 * @param vecPoint 获取人脸五个点
 */
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


/**
 * 执行单次单人的facenet网络
 * @param image 输入图片
 * @param vecRect 人脸框
 * @param n emb值
 */
void run_facenet(Mat &image, vector<Rect> &vecRect, vector<mydataFmt> &n) {
    Mat fourthImage;
    resize(image(vecRect[0]), fourthImage, Size(160, 160), 0, 0, cv::INTER_LINEAR);
    facenet ggg;
//        mydataFmt *o = new mydataFmt[Num];
//    vector<mydataFmt> n;
//    vector<vector<mydataFmt>> o;
    ggg.run(fourthImage, n, 0);
}

/**
 * 对比两张图两个人的emb
 */
void run() {
    Mat image0 = imread("../test_img/tom0.jpeg");
    Mat image1 = imread("../test_img/tom1.jpeg");
    //缩放一下图像
    image0 = RS(image0);
    image1 = RS(image1);

    clock_t start;
    start = clock();
    vector<Rect> vecRect0, vecRect1;
    vector<mydataFmt> n0, n1;
    vector<Point> vecPoint0, vecPoint1;
    run_mtcnn(image0, vecRect0, vecPoint0);
    run_facenet(image0, vecRect0, n0);
    run_mtcnn(image1, vecRect1, vecPoint1);
    run_facenet(image1, vecRect1, n1);

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

    float result = compare(n0, n1);
    cout << "-------------------" << endl;
    cout << result << endl;
    if (result < 0.45)
        cout << "Probably the same person" << endl;
    else
        cout << "Probably not the same person" << endl;

    imshow("result0", image0);
//    resizeWindow("result0", w0, h0); //创建一个固定值大小的窗口
    imwrite("../test_img/result0.jpg", image0);
    imshow("result1", image1);
    imwrite("../test_img/result1.jpg", image1);
    start = clock() - start;
    //    cout<<"time is  "<<start/10e3<<endl;
    cout << "time is " << (double) start / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    waitKey(0);
    image0.release();
    image1.release();
}

/**
 * 程序主体
 * @return
 */
int main() {
    for (int i = 0; i < 1; ++i) {
        run();
        cout << "==============================" << endl;
    }
    return 0;
}
