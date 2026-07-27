// OpenCV_ExtensionTool.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//
#pragma once
#include <iostream>

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp> //mophorlogical operation
#include<opencv2/core.hpp>
#include <numeric>
#include "OpenCV_Extension_Tool.h"
#include "OpenCV_DEBUG_Tool.h"
#include "MatchTool.h"
#include "MaterialMatcher.h"

using namespace cv;
using namespace std;


int main()
{

    Mat imgSample = imread("C:\\Image\\光斑 LEVEL\\擷取DWDWDFWDWDDW.bmp");

    //vector<Mat> channels;

    //split(imgSample, channels);

    //Mat imgSampleB = channels[0];
    //Mat imgSampleG = channels[1];
    //Mat imgSampleR = channels[2];




    //讀取光斑數據
    std::unordered_map<double, Mat> dict;
	vector<double> vLightSpotLv;

	string strFilePath = "C:\\Image\\光斑 LEVEL\\SampleSET_RGB";


	vector<Size> vSize;
    std::unordered_map<double, vector<s_SingleTargetMatch>> dicMatchedResult;
    std::unordered_map<double, vector<s_SingleTargetMatch>> dicMatchedResult_R;
    std::unordered_map<double, vector<s_SingleTargetMatch>> dicMatchedResult_G;
    std::unordered_map<double, vector<s_SingleTargetMatch>> dicMatchedResult_B;

    //std::unordered_map<double, double> dicCompactness;
    vector<double> vCompactness;

    double dMax = -999999;
    double dMin = 999999;

    MaterialMatcher matcher;

    //----Load DB
    std::string dbPath = "C:\\Image\\光斑 LEVEL\\LightSpotDB\\";

    std::vector<cv::Mat> samples;
    samples.push_back(cv::imread(dbPath+"60-1.bmp"));
    samples.push_back(cv::imread(dbPath+"60-2.bmp"));
    samples.push_back(cv::imread(dbPath+"60-3.bmp"));
    matcher.BuildMaterial(60,samples);

    std::vector<cv::Mat> samples2;
    samples2.push_back(cv::imread(dbPath+"100-1.bmp"));
    samples2.push_back(cv::imread(dbPath+"100-2.bmp"));
    samples2.push_back(cv::imread(dbPath+"100-3.bmp"));
    matcher.BuildMaterial(100, samples2);




    for(int i = 1; i <= 10; i++)
    {
		int dLv = (int)i * 10;
        string strFileName = strFilePath + "\\" + to_string(dLv) + ".bmp";
        Mat img = imread(strFileName);
		vSize.push_back(img.size());
        dict.insert({ dLv, img });
		dicMatchedResult.insert({ dLv, vector<s_SingleTargetMatch>() });
        vLightSpotLv.push_back(dLv);

            //    "Aluminum",
            //    samples);
	}


    //---建立計算公式

    //= 90 * R16(LN(R5) - LN(186.57)) / (LN(1012.67) - LN(186.57))

    //double result = 90.0  *
    //    (std::log(dCompactness) - std::log(campactnessMin)) /
    //    (std::log(campactnessMax) - std::log(campactnessMin));


    //計算平均大小
    double avgWidth = 0;
	double avgHeight = 0;
    
	for (const auto& sz : vSize)
	{
		avgWidth += sz.width;
		avgHeight += sz.height;
	}

	avgWidth /= vSize.size();
	avgHeight /= vSize.size();

    //----方便分區
	cout << "Average Width: " << avgWidth << endl;
	cout << "Average Height: " << avgHeight << endl;


    for (int i = 10; i <= 100; i+=10)
    {


        CMatchTool matchTool = CMatchTool();
        matchTool.LearnPattern(dict[i], 50, 0.7, 5, 0.5, 100);
        vector<s_SingleTargetMatch> result;
        matchTool.Match(imgSample, result);

        Mat gray1;
        cvtColor(imgSample, gray1, COLOR_BGR2GRAY);


        //dicMatchedResult[i] = result;

        Mat tmp1 = imgSample.clone();

        if (result.size() > 0)
        {
            putText(tmp1, to_string(i), Point(100,500), FONT_HERSHEY_COMPLEX, 20, Scalar(0, 0, 255), 2, 1);


            for(int j = 0; j < result.size(); j++)
            {
                Point2f ptC = result[j].ptCenter;
                String str = to_string(result[j].dMatchScore);
                //putText(tmp1, str, ptC, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, 1);



                vector<double> vX;
                vector<double> vY;

                vX.push_back(result[j].ptLB.x);
                vX.push_back(result[j].ptLT.x);
                vX.push_back(result[j].ptRB.x);
                vX.push_back(result[j].ptRT.x);

                vY.push_back(result[j].ptLB.y);
                vY.push_back(result[j].ptLT.y);
                vY.push_back(result[j].ptRB.y);
                vY.push_back(result[j].ptRT.y);

                //cv::Mat image = cv::imread("test.jpg");

                double minValueX = *std::min_element(vX.begin(), vX.end());
                double maxValueX = *std::max_element(vX.begin(), vX.end());

                double minValueY = *std::min_element(vY.begin(), vY.end());
                double maxValueY = *std::max_element(vY.begin(), vY.end());

                //// x, y, width, height
                cv::Rect roi((int)minValueX, (int)minValueY,(int)(maxValueX-minValueX), (int)(maxValueY - minValueY));

                cv::Mat crop = gray1(roi);

                MaterialMatchResult resultMM =
                    matcher.Match(crop);
                if (resultMM.Found)
                {
                    std::cout
                        << "Material : "
                        << resultMM.Name
                        << std::endl;

                    std::cout
                        << "Similarity : "
                        << resultMM.Score
                        << std::endl;

                    putText(tmp1, resultMM.Name, ptC, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, 1);
                }
                else
                {
                    std::cout
                        << "No Match"
                        << std::endl;
                }
			}
            









        }


    }








	system("pause");

    //vector<string> vStr;
    //GetAllFolderBmpImage("C:\\Git\\Code\\OpenCV_Tool\\OpenCV_ExtensionTool", vStr);

    //cvtColor(imgSample, ttt, COLOR_RGB2GRAY, 1);

    //threshold(ttt, ttt, 150, 255, THRESH_BINARY);

    //------測試 BLOB

    int gray = 10;

    //auto TimeStart = std::chrono::high_resolution_clock::now();
    //
    //BlobFilter b_Filter = BlobFilter();

    //b_Filter.SetEnableArea(false);
    //b_Filter.SetMaxArea(20000);
    //b_Filter.SetMinArea(100);

    ////將所有連通區域切割 並萃取各區域的屬性
    ////vector<BlobInfo> lst= RegionPartition(ttt,INT16_MAX,0);
    ////b_Filter.~BlobFilter();
    //vector<BlobInfo> lst = RegionPartition(ttt);

    //auto TimeEnd = std::chrono::high_resolution_clock::now();

    //double countingTime = std::chrono::duration<double, std::milli>(TimeEnd - TimeStart).count();
    //std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;
    //std::cout << "calculate countingTime time is:: " << countingTime << endl;
    //std::cout << "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*" << endl;

    //ShowDebugWindow(imgXXXX, lst);

    //cvtColor(ttt, ttt, COLOR_GRAY2RGB);


    //for (size_t i = 0; i < lst.size(); i++)
    //    circle(ttt, lst[i].Center(), 2, Scalar(0, 0, 255), 1);


    //已測試 切割 210個 Region
    //純粹用洪水法分區 26ms
    // 使用 RegionFloodFill  3ms 效率提升 8倍
    // 存取指標方式 14ms 反而比較慢 後續不採用
    // 只要有創建新影像 有進行影像操作 速度就快不起來


    // 0808 測試
    // RegionPartition(Mat ImgBinary, int maxArea, int minArea) 改成全指標方式存取 急速可以撐到 9.5ms 速度會在9.5~14之間飄移
    // 影響速度變化的原因待查

}


