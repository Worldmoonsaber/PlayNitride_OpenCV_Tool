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

using namespace cv;
using namespace std;



int main()
{
    Mat imgXXXX = imread("C:\\Git\\Code\\OpenCV_Tool\\OpenCV_ExtensionTool\\test.jpg");
    Mat ttt;

    cvtColor(imgXXXX, ttt, COLOR_RGB2GRAY, 1);
    threshold(ttt, ttt, 150, 255, THRESH_BINARY);

    //------測試 BLOB

    int gray = 10;
    auto TimeStart = std::chrono::high_resolution_clock::now();
    
    BlobFilter b_Filter = BlobFilter();

    b_Filter.SetEnableArea(false);
    b_Filter.SetMaxArea(20000);
    b_Filter.SetMinArea(100);

    //將所有連通區域切割 並萃取各區域的屬性
    //vector<BlobInfo> lst= RegionPartition(ttt,INT16_MAX,0);
    //b_Filter.~BlobFilter();
    //vector<BlobInfo> lst = RegionPartition(ttt);

    //vector<BlobInfo> lst = RegionPartition(ttt, b_Filter);
    vector<BlobInfo> lst = RegionPartitionTopology(ttt, b_Filter);
    auto TimeEnd = std::chrono::high_resolution_clock::now();

    double countingTime = std::chrono::duration<double, std::milli>(TimeEnd - TimeStart).count();
    std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
    std::cout << "calculate countingTime time is:: " << countingTime << endl;
    std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;

    ShowDebugWindow(imgXXXX, lst);
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
}


