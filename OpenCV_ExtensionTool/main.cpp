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
#include "MatchToolDlg.h"

using namespace cv;
using namespace std;



int main()
{
    //Mat imgXXXX = imread("C:\\Git\\ResolutionCaculator\\WinFormResolCalibrator\\WinFormResolCalibrator\\bin\\Debug\\net8.0-windows7.0\\Debug_10-01-59.bmp");
    //Mat ttt;

    //Mat imgXXXX = imread("C:\\Image\\Uchip\\L5\\20230613\\62001.bmp");
    //Mat rawimg = imread("C:\\Image\\Pair Chip\\20240830 PN177 chips image\\3_I140.bmp");

    //Mat Gimg;
    //Mat ImgThres,ImgThres2;
    //cv::cvtColor(rawimg, Gimg, COLOR_RGB2GRAY);

    //adaptiveThreshold(Gimg, ImgThres, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 101, 0);//55,1 //ADAPTIVE_THRESH_MEAN_C
    //adaptiveThreshold(Gimg, ImgThres2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 101, 0);//55,1 //ADAPTIVE_THRESH_MEAN_C

    ////cv::medianBlur(adptThres, ImgThres, 7);
    //Mat Kcomopen = Mat::ones(Size(10, 10), CV_8UC1);  //Size(10,5)
    //Mat Kcomclose = Mat::ones(Size(5, 5), CV_8UC1);  //Size(10,5)
    //
    //cv::morphologyEx(ImgThres, ImgThres, cv::MORPH_OPEN, Kcomopen, Point(-1, -1), 1);//1 //2
    //cv::morphologyEx(ImgThres, ImgThres, cv::MORPH_CLOSE, Kcomopen, Point(-1, -1), 1);//1 //2

    //cv::morphologyEx(ImgThres2, ImgThres2, cv::MORPH_OPEN, Kcomopen, Point(-1, -1), 1);//1 //2
    //cv::morphologyEx(ImgThres2, ImgThres2, cv::MORPH_CLOSE, Kcomopen, Point(-1, -1), 1);//1 //2

    //
    //Mat merge = ImgThres + ImgThres2;
    //Mat Kcomclose2 = Mat::ones(Size(3, 3), CV_8UC1);  //Size(10,5)
    //cv::morphologyEx(merge, merge, cv::MORPH_CLOSE, Kcomclose2, Point(-1, -1), 1);//1 //2

    //auto TimeStart = std::chrono::high_resolution_clock::now();
    //vector<BlobInfo> lst = RegionPartitionTopology(merge);
    //auto TimeEnd = std::chrono::high_resolution_clock::now();


    //vector<BlobInfo> lstU;

    //for (int u = 0; u < lst.size(); u++)
    //{
    //    if(lst[u].Width()>100)
    //        lstU.push_back(lst[u]);
    //}




    //vector<vector<Point>> vContour;
    //for (int u = 0; u < lstU.size(); u++)
    //{
    //    vContour.push_back(lstU[u].contour());
    //}


    //drawContours(merge, vContour, -1, Scalar(100, 100, 100), -1);



    //double countingTime = std::chrono::duration<double, std::milli>(TimeEnd - TimeStart).count();
    //std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
    //std::cout << "calculate countingTime time is:: " << countingTime << endl;
    //std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;


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

    CMatchToolDlg matchTest = CMatchToolDlg();

    matchTest.m_matSrc= imread("C:\\Git\\OpenCV_Tool\\TEST\\Match_Test_Sample5.bmp");
    matchTest.m_matDst = imread("C:\\Git\\OpenCV_Tool\\TEST\\790301_chip.bmp");
    matchTest.m_dScore = 0.5;
    matchTest.m_dToleranceAngle = 0;// 20;
    matchTest.m_dMaxOverlap = 0;
    matchTest.m_iMinReduceArea = 256;
    matchTest.LearnPattern();
    matchTest.Match();
    //Mat img = imread("C:\\Git\\OpenCV_Tool\\TEST\\790301.bmp");
    //Mat imgPattern = imread("C:\\Git\\OpenCV_Tool\\TEST\\790301_chip.bmp");

    //s_TemplData m_TemplData;
    //LearnPattern(imgPattern, m_TemplData);

    //match_Param pm;

    //pm.AngleTolerance = 40;
    //pm.MaxiumResults = 999;
    //pm.Score = 0.5;


    //vector<s_SingleTargetMatch> resultPt;
    //Match(img, imgPattern, pm, m_TemplData, resultPt);

    Mat img=imread("C:\\Git\\OpenCV_Tool\\TEST\\Match_Test_Sample5.bmp");
    Mat pattern = imread("C:\\Git\\OpenCV_Tool\\TEST\\790301_chip.bmp");


    for (int i = 0; i < matchTest.m_vecSingleTargetData.size(); i++)
    {
        Point2f pt1[4] = { matchTest.m_vecSingleTargetData[i].ptLT ,
                            matchTest.m_vecSingleTargetData[i].ptRT,
                            matchTest.m_vecSingleTargetData[i].ptRB,
                            matchTest.m_vecSingleTargetData[i].ptLB};




        line(img, pt1[0] , pt1[1] , Scalar(20, 20, 255), 5);
        line(img, pt1[1] , pt1[2] , Scalar(20, 20, 255), 5);
        line(img, pt1[2] , pt1[3] , Scalar(20, 20, 255), 5);
        line(img, pt1[3] , pt1[0] , Scalar(20, 20, 255), 5);
    }




    //Mat trans2 = getPerspectiveTransform(pt1, pt0, DECOMP_LU);

    //Mat result2;
    //warpPerspective(img, result2, trans2, imgPattern.size()); //將Pattern 映射到對應位置




    system("pause");
}


