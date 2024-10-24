// OpenCV_ExtensionTool.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//

#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp> //mophorlogical operation
#include<opencv2/core.hpp>
#include <numeric>
#include "OpenCV_Extension_Tool.h"
#include "OpenCV_DEBUG_Tool.h"
#include "MatchTool.h"

using namespace cv;
using namespace std;



int main()
{

    CMatchTool matchTest = CMatchTool();


    Mat pattern = imread("C:\\Git\\OpenCV_Tool\\TEST\\KeySample.bmp");
    Mat img = imread("C:\\Git\\OpenCV_Tool\\TEST\\20240919_121620.bmp");


    auto t_start = std::chrono::high_resolution_clock::now();


    vector<s_SingleTargetMatch> result;

    matchTest.LearnPattern(pattern, 1, 0.4, 20, 0, 256);
    matchTest.SetMatchConfig(false, false);

    matchTest.Match(img, result);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "match time is:: " << elapsed_time_ms << endl;
    std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;


    for (int i = 0; i < result.size(); i++)
    {
        Point2f pt1[4] = { result[i].ptLT ,
                            result[i].ptRT,
                            result[i].ptRB,
                            result[i].ptLB};

        line(img, pt1[0] , pt1[1] , Scalar(20, 20, 255), 2,1);
        line(img, pt1[1] , pt1[2] , Scalar(20, 20, 255), 2, 1);
        line(img, pt1[2] , pt1[3] , Scalar(20, 20, 255), 2, 1);
        line(img, pt1[3] , pt1[0] , Scalar(20, 20, 255), 2, 1);

        Point2f pt1Mark = result[i].ptLT * 0.8 + 0.2 * result[i].ptRT;
        Point2f pt2Mark = result[i].ptLT * 0.8 + 0.2 * result[i].ptLB;
        line(img, pt1Mark, pt2Mark, Scalar(100, 255, 100), 2, 20);
    }

    system("pause");
}