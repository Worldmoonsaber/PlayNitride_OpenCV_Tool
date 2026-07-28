// OpenCV_ExtensionTool.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//
#pragma once

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "MaterialProfile.h"
#include "MatchTool.h"
#include "OpenCV_Extension_Tool.h"
#include "OpenCV_DEBUG_Tool.h"

using namespace cv;
using namespace std;

namespace
{

// 僅供建模導入使用；建立新產品模型時修改這些數值並保留在 main。
// Commissioning only; keep these values in main and update them for a new product.
MaterialProductRecipe CreateProductRecipe()
{
    constexpr std::array<int, 7> columnCenters = {
        436, 1060, 1668, 2272, 2880, 3484, 4088};
    constexpr std::array<int, 10> rowCenters = {
        864, 1212, 1552, 1892, 2236,
        2572, 2916, 3260, 3600, 3940};
    constexpr std::array<int, 4> trainingColumnIndices = {0, 2, 4, 6};
    constexpr std::array<const char*, 4> labels = {"40", "60", "80", "100"};
    constexpr int roiWidth = 420;
    constexpr int roiHeight = 250;

    MaterialProductRecipe recipe;
    recipe.productId = "DefaultProduct";
    recipe.standardImagePath =
        L"C:\\Image\\\u5149\u6591 LEVEL\\"
        L"\u64f7\u53d6DWDWDFWDWDDW.bmp";
    recipe.databaseDirectory =
        L"C:\\Image\\\u5149\u6591 LEVEL\\MaterialProfiles";

    // 修改建模設定後設為 true 執行一次，產線使用時改回 false 載入既有資料庫。
    // Set true once after changing commissioning settings; restore false for production loading.
    recipe.forceRetrain = false;
    recipe.recognizerOptions.feature.enableTextureFeatures = false;
    recipe.recognizerOptions.feature.maxImageSide = 1024;
    recipe.recognizerOptions.rejectConfidence = 0.40;
    recipe.recognizerOptions.enablePca = false;
    recipe.recognizerOptions.covarianceShrinkage = 1.0;
    recipe.recognizerOptions.featureWeights["Color"] = 1.0;

    for (std::size_t classIndex = 0; classIndex < labels.size(); ++classIndex)
    {
        const int centerX = columnCenters[trainingColumnIndices[classIndex]];
        for (const int centerY : rowCenters)
        {
            recipe.trainingRois.push_back({
                labels[classIndex],
                cv::Rect(
                    centerX - roiWidth / 2,
                    centerY - roiHeight / 2,
                    roiWidth,
                    roiHeight)});
        }
    }
    return recipe;
}

} // namespace

int main()
{
    MaterialProductRecipe materialRecipe = CreateProductRecipe();
    MaterialProfile materialProfile(materialRecipe);
    materialProfile.Initialize();

    cv::Mat imgSample = imread("C:\\Image\\光斑 LEVEL\\TEST SAMPLE\\60-C.bmp");// materialProfile.LoadStandardImage();

    //讀取光斑數據
    std::unordered_map<double, Mat> dict;
    string strFilePath =
        "C:\\Image\\光斑 LEVEL\\SampleSET_RGB";
    vector<Size> vSize;

    for(int i = 1; i <= 10; i++)
    {
		int dLv = (int)i * 10;
        string strFileName = strFilePath + "\\" + to_string(dLv) + ".bmp";
        Mat img = imread(strFileName);
		vSize.push_back(img.size());
        dict.insert({ dLv, img });
	}


    for (int i = 10; i <= 100; i+=10)
    {


        CMatchTool matchTool = CMatchTool();
        matchTool.LearnPattern(dict[i], 50, 0.7, 5, 0.5, 100);
        vector<s_SingleTargetMatch> result;
        matchTool.Match(imgSample, result);

        Mat gray1;
        cvtColor(imgSample, gray1, COLOR_BGR2GRAY);

        Mat tmp1 = imgSample.clone();

        if (result.size() > 0)
        {
            putText(tmp1, to_string(i), Point(100,500), FONT_HERSHEY_COMPLEX, 20, Scalar(0, 0, 255), 2, 1);


            for(int j = 0; j < result.size(); j++)
            {
                Point2f ptC = result[j].ptCenter;
                String str = to_string(result[j].dMatchScore);

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

                double minValueX = *std::min_element(vX.begin(), vX.end());

                if (minValueX < 0)
                    minValueX = 0;

                double maxValueX = *std::max_element(vX.begin(), vX.end());

                if (maxValueX >= imgSample.cols)
                    maxValueX = imgSample.cols - 1;


                double minValueY = *std::min_element(vY.begin(), vY.end());

                if (minValueY < 0)
                    minValueY = 0;

                double maxValueY = *std::max_element(vY.begin(), vY.end());

                if (maxValueY>= imgSample.rows)
                    maxValueY = imgSample.rows - 1;



                cv::Rect roi((int)minValueX, (int)minValueY,(int)(maxValueX-minValueX), (int)(maxValueY - minValueY));
                cv::Mat crop = imgSample(roi);

                const material::RecognitionResult materialResult =
                    materialProfile.Recognize(crop);

                if (materialResult.accepted)
                {
                    cout << "----------------------------" << endl;
                    cout << "Material   : "
                        << materialResult.label << endl;
                    cout << "Confidence : "
                        << materialResult.confidence << endl;
                    cout << "Distance   : "
                        << materialResult.distance << endl;

                    for (const auto& candidate :
                         materialResult.candidates)
                    {
                        cout
                            << candidate.label
                            << "  confidence="
                            << candidate.confidence
                            << "  distance="
                            << candidate.distance
                            << endl;
                    }

                    cout << "----------------------------" << endl;

                    const int materialLevel =
                        std::stoi(materialResult.label);

                    putText(
                        tmp1,
                        to_string(materialLevel),
                        ptC,
                        FONT_HERSHEY_COMPLEX,
                        1,
                        Scalar(0, 0, 255),
                        2,
                        1);
                }
                else
                {
                    cout
                        << "Unknown material. Best candidate="
                        << materialResult.label
                        << " confidence="
                        << materialResult.confidence
                        << endl;
                }
			}
            









        }

        system("pause");
    }

	system("pause");

}


