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
    //ShowDebugWindow(imgXXXX, lst);
    ////已測試 切割 210個 Region
    ////純粹用洪水法分區 26ms
    // 使用 RegionFloodFill  3ms 效率提升 8倍
    // 存取指標方式 14ms 反而比較慢 後續不採用
    // 只要有創建新影像 有進行影像操作 速度就快不起來


    // 0808 測試
    // RegionPartition(Mat ImgBinary, int maxArea, int minArea) 改成全指標方式存取 急速可以撐到 9.5ms 速度會在9.5~14之間飄移
    // 影響速度變化的原因待查

    //  0815 拓樸學實現
    //
    // 0909 拓樸學開發出花費時間 在可以接受範圍的實作結果

    //---Match 測試

    CMatchTool matchTest = CMatchTool();


    Mat pattern = imread("C:\\Git\\OpenCV_Tool\\TEST\\790301_chip.bmp");
    Mat img = imread("C:\\Git\\OpenCV_Tool\\TEST\\Match_Test_Sample5.bmp");

    auto t_start = std::chrono::high_resolution_clock::now();


    vector<s_SingleTargetMatch> result;

    matchTest.LearnPattern(pattern, 500, 0.7, 20, 0, 1024);
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

        line(img, pt1[0] , pt1[1] , Scalar(20, 20, 255), 5);
        line(img, pt1[1] , pt1[2] , Scalar(20, 20, 255), 5);
        line(img, pt1[2] , pt1[3] , Scalar(20, 20, 255), 5);
        line(img, pt1[3] , pt1[0] , Scalar(20, 20, 255), 5);
    }

    system("pause");
}


