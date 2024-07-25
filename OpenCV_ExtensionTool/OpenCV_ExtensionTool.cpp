// OpenCV_ExtensionTool.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//

#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp> //mophorlogical operation
#include<opencv2/core.hpp>
#include <numeric>
#include "OpenCV_Extension_Tool.h"


using namespace cv;
using namespace std;


bool mouseClicked = false;

static void onMouse(int event, int x, int y, void* userInput)
{

}



int main()
{
    /*
    
    Area ,Circularity, Convexity,Inertia 
    可以單純用Blob處理 但是此處只會抓出符合條件區域的中心點
    且沒有其他可以偵測的特徵


    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 1500;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.1;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;

    */






    std::cout << "Hello World!\n";
    //測試程式碼

    Mat imgXXXX;

    imgXXXX = imread("test3.jpg");
    Mat ttt;
//    imgXXXX = imread("C:\\Sample Image\\TestSample_SR1\\13.bmp");



    cvtColor(imgXXXX, imgXXXX, COLOR_RGB2GRAY, 1);

    GaussianBlur(imgXXXX, imgXXXX, Size(3, 3), 1, 1);

    threshold(imgXXXX, ttt, 200, 255, THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_RECT, Size(5,5));

    cv::morphologyEx(ttt, ttt, MORPH_CLOSE, element, Point(-1, -1));



    //------測試 BLOB

    int gray = 10;

    auto TimeStart = std::chrono::high_resolution_clock::now();
    
    //將所有連通區域切割 並萃取各區域的屬性
    vector<BlobInfo> lst= RegionPartition(ttt,10000);

    auto TimeEnd = std::chrono::high_resolution_clock::now();

    double countingTime = std::chrono::duration<double, std::milli>(TimeEnd - TimeStart).count();
    std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
    std::cout << "calculate countingTime time is:: " << countingTime << endl;
    std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;

    std::cout << "請按任意件切換選擇的區域" << countingTime << endl;


    for (size_t i = 0; i < lst.size(); i++)
    {
        Mat Demo = ttt.clone();

        for (int j = 0; j < lst[i].Points().size(); j++)
            Demo.at<uchar>(lst[i].Points()[j].y, lst[i].Points()[j].x) = 100;

        imshow("Img", Demo);

        cv::waitKey(0);
    }







    //已測試 切割 210個 Region
    //純粹用洪水法分區 26ms
    // 使用 RegionFloodFill  3ms 效率提升 8倍
    // 存取指標方式 14ms 反而比較慢 後續不採用
    // 只要有創建新影像 有進行影像操作 速度就快不起來

}


