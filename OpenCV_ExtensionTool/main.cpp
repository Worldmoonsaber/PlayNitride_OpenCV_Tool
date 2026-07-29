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
        // 正常訓練資料結構：
        // TrainingSample\<標籤>\*.bmp，例如 TrainingSample\40\40-1.bmp。
        // Normal layout:
        // TrainingSample\<label>\*.bmp, for example TrainingSample\40\40-1.bmp.
        const std::filesystem::path trainingRoot =
            L"C:\\Image\\\u5149\u6591 LEVEL\\TrainingSample";

        // 修改訓練資料後設為 true 執行一次；模型建立後，上線改回 false。
        // Set true once after changing samples; switch back to false in production.
        constexpr bool forceRetrain = true;

        const std::set<std::wstring> supportedExtensions = {
            L".bmp", L".png", L".jpg", L".jpeg", L".tif", L".tiff" };

        if (!std::filesystem::is_directory(trainingRoot))
        {
            throw std::runtime_error(
                "TrainingSample directory does not exist: " +
                trainingRoot.string());
        }

        MaterialProductRecipe recipe;
        recipe.databasePath =
            L"C:\\Image\\\u5149\u6591 LEVEL\\MaterialProfiles\\"
            L"TrainingSampleProduct_ColorV2_new.yml";
        recipe.forceRetrain = forceRetrain;
        //recipe.recognizerOptions.feature.enableTextureFeatures = false;
        //recipe.recognizerOptions.feature.maxImageSide = 1024;
        //recipe.recognizerOptions.rejectConfidence = 0.40;
        //recipe.recognizerOptions.enablePca = false;
        //recipe.recognizerOptions.covarianceShrinkage = 1.0;
        //recipe.recognizerOptions.featureWeights["Color"] = 1.0;

        for (const std::filesystem::directory_entry& labelDirectory :
            std::filesystem::directory_iterator(trainingRoot))
        {
            if (!labelDirectory.is_directory())
            {
                continue;
            }

            const std::string label =
                labelDirectory.path().filename().string();
            std::size_t labelSampleCount = 0;
            for (const std::filesystem::directory_entry& sampleFile :
                std::filesystem::directory_iterator(labelDirectory.path()))
            {
                if (!sampleFile.is_regular_file())
                {
                    continue;
                }

                std::wstring extension =
                    sampleFile.path().extension().wstring();
                std::transform(
                    extension.begin(),
                    extension.end(),
                    extension.begin(),
                    ::towlower);
                if (supportedExtensions.count(extension) != 0)
                {
                    recipe.trainingImages.push_back(
                        { label, sampleFile.path() });
                    ++labelSampleCount;
                }
            }

            std::cout
                << "Training label " << label
                << ": " << labelSampleCount << " images\n";
        }

        if (recipe.trainingImages.empty())
        {
            throw std::runtime_error(
                "No training images were found under TrainingSample.");
        }

        std::cout
            << "Training source: " << trainingRoot.string() << '\n'
            << "Database path : " << recipe.databasePath.string() << '\n'
            << "Model mode    : "
            << (recipe.forceRetrain
                ? "retrain from TrainingSample"
                : "load database; train only when database is missing")
            << '\n';
        return recipe;
    }


    double Decay(double score)
    {
        return std::pow(score, 3.0);
    }

} // namespace

int main()
{
    //-----讀取已有的訓練模型
    //MaterialProductRecipe materialRecipe;
    //materialRecipe.databasePath =
    //    L"C:\\Image\\\u5149\u6591 LEVEL\\MaterialProfiles\\TrainingSampleProduct_ColorV2.yml";
    //materialRecipe.forceRetrain = false;
    //MaterialProfile materialProfile(materialRecipe);
    //materialProfile.Initialize();

	//-----讀取光斑影像
    MaterialProductRecipe materialRecipe= CreateProductRecipe();
    MaterialProfile materialProfile(materialRecipe);
    materialProfile.Initialize();


    
    cv::Mat imgSample = imread("C:\\Image\\光斑 LEVEL\\擷取DWDWDFWDWDDW.bmp");

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

                    double dConfidenceWeight = 0;
					double dConfidenceSum = 0;

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

                        dConfidenceWeight += Decay(candidate.confidence) * std::stoi(candidate.label);
                        dConfidenceSum += Decay(candidate.confidence);
                    }

                    cout << "----------------------------" << endl;

                    const int materialLevel =
                        std::stoi(materialResult.label);

                    int nWeightedLevel = (int)(dConfidenceWeight / dConfidenceSum);

                    cout
                        << "  nWeightedLevel="
                        << nWeightedLevel
                        << endl;

                    cout << "***************************" << endl;

                    putText(
                        tmp1,
                        to_string(nWeightedLevel),
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

	//system("pause");

}


