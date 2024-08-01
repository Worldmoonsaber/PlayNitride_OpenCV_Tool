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
    std::cout << "Hello World!\n";
    //測試程式碼

    Mat imgXXXX;

    imgXXXX = imread("C:\\Sample Image\\StampChip\\Cplus\\Stp0718\\7180127.bmp");
    Mat ttt;

    cvtColor(imgXXXX, ttt, COLOR_RGB2GRAY, 1);

    threshold(ttt, ttt, 100, 255, THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_RECT, Size(50,50));

    cv::morphologyEx(ttt, ttt, MORPH_CLOSE, element, Point(-1, -1));

    //------測試 BLOB

    int gray = 10;

    auto TimeStart = std::chrono::high_resolution_clock::now();
    

    BlobFilter b_Filter = BlobFilter();

    b_Filter.SetEnableArea(true);
    b_Filter.SetMaxArea(200*100*1.5);
    b_Filter.SetMinArea(200 * 100 * 0.1);

    b_Filter.SetEnableXbound(true);
    b_Filter.SetMaxXbound(3000);
    b_Filter.SetMinXbound(2500);

    b_Filter.SetEnableYbound(true);
    b_Filter.SetMaxYbound(2000);
    b_Filter.SetMinYbound(1000);


    //將所有連通區域切割 並萃取各區域的屬性
    vector<BlobInfo> lst= RegionPartition(ttt, b_Filter);
    //b_Filter.~BlobFilter();

    auto TimeEnd = std::chrono::high_resolution_clock::now();

    double countingTime = std::chrono::duration<double, std::milli>(TimeEnd - TimeStart).count();
    std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
    std::cout << "calculate countingTime time is:: " << countingTime << endl;
    std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;

    std::cout << "請按任意件切換選擇的區域" << countingTime << endl;


    for (size_t i = 0; i < lst.size(); i++)
    {
        Mat Demo = imgXXXX.clone();


        circle(Demo, lst[i].Center(), 100, Scalar(0, 0, 255), 5);

        imshow("Img", Demo);

        cv::waitKey(0);
    }







    //已測試 切割 210個 Region
    //純粹用洪水法分區 26ms
    // 使用 RegionFloodFill  3ms 效率提升 8倍
    // 存取指標方式 14ms 反而比較慢 後續不採用
    // 只要有創建新影像 有進行影像操作 速度就快不起來

}


