#include "network.h"
#include "mtcnn.h"
#include "facenet.h"
#include <time.h>

void run() {
    int b = 0;
    if (b == 0) {
        Mat image = imread("../40.jpg");
//        Mat image = imread("../1.jpeg");
//        Mat image = imread("../Kong_Weiye.jpg");
//        Mat image = imread("../Kong_Weiye1.jpg");
//        Mat image = imread("../20.png");
//        Mat image = imread("../emb_img/0.jpg");
        mtcnn find(image.rows, image.cols);
        clock_t start;
        start = clock();
        find.findFace(image);
        imshow("result", image);
        imwrite("../result.jpg", image);
        start = clock() - start;
        //    cout<<"time is  "<<start/10e3<<endl;
        cout << "time is " << (double) start / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        waitKey(0);
        image.release();
    } else if (b == 1) {
        Mat image = imread("../10.jpg");
//        Mat image = imread("../emb_img/0.jpg");
//        Mat image = imread("../20.png");
        Mat Image;
        resize(image, Image, Size(160, 160), 0, 0, cv::INTER_LINEAR);
        facenet ggg;
        mydataFmt *o = new mydataFmt[Num];
        ggg.run(Image, o, 0);
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
            find.findFace(image);
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
