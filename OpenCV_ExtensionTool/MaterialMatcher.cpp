#include "MaterialMatcher.h"

using namespace cv;
using namespace std;

bool MaterialMatcher::BuildMaterial(
    int nLightIntensity,
    const vector<Mat>& sampleImages)
{
    if (sampleImages.empty())
        return false;

    MaterialFeature material;

    material.LightIntensity = nLightIntensity;
    material.SampleCount =
        static_cast<int>(sampleImages.size());

    vector<float> sumFeature;

    for (const auto& img : sampleImages)
    {
        vector<float> feature =
            ExtractFeature(img);

        if (feature.empty())
            continue;

        if (sumFeature.empty())
        {
            sumFeature.resize(
                feature.size(),
                0.0f);
        }

        for (size_t i = 0; i < feature.size(); i++)
        {
            sumFeature[i] += feature[i];
        }
    }

    if (sumFeature.empty())
        return false;

    for (float& value : sumFeature)
    {
        value /= sampleImages.size();
    }

    material.GrayFeature = std::move(sumFeature);

    m_Database.push_back(material);

    return true;
}

void MaterialMatcher::AddMaterial(
    const MaterialFeature& material)
{
    m_Database.push_back(material);
}

void MaterialMatcher::ClearDatabase()
{
    m_Database.clear();
}

const vector<MaterialFeature>&
MaterialMatcher::GetDatabase() const
{
    return m_Database;
}

MaterialMatchResult MaterialMatcher::Match(const cv::Mat& image)
{
    MaterialMatchResult result;

    if (m_Database.empty())
        return result;

    auto feature =
        ExtractFeature(image);

    double bestScore = -1.0;

    int bestIndex = -1;

    for (size_t i = 0; i < m_Database.size(); i++)
    {
        double score =
            CosineSimilarity(
                feature,
                m_Database[i].GrayFeature);

        result.dicResultScores.insert(
			{ m_Database[i].LightIntensity, score });

        if (score > bestScore)
        {
            bestScore = score;
            bestIndex = (int)i;
        }
    }

    if (bestIndex >= 0)
    {
        result.Found = true;
        result.Index = bestIndex;
        result.Name = to_string(m_Database[bestIndex].LightIntensity);
        result.Score = bestScore;
    }

    return result;
}

vector<float>
MaterialMatcher::ExtractFeature(
    const Mat& image)
{
    vector<float> feature;

    if (image.empty())
        return feature;

    Mat gray;

    if (image.channels() == 3)
    {
        cvtColor(
            image,
            gray,
            COLOR_BGR2GRAY);
    }
    else
    {
        gray = image.clone();
    }

    //-----------------------------------------
    // Mean + Std
    //-----------------------------------------

    Scalar meanValue;
    Scalar stdValue;

    meanStdDev(
        gray,
        meanValue,
        stdValue);

    feature.push_back(
        (float)meanValue[0]);

    feature.push_back(
        (float)stdValue[0]);

    //-----------------------------------------
    // Histogram
    //-----------------------------------------

    int histSize = 32;

    float range[] =
    {
        0,
        256
    };

    const float* histRange =
    {
        range
    };

    Mat hist;

    calcHist(
        &gray,
        1,
        0,
        Mat(),
        hist,
        1,
        &histSize,
        &histRange);

    normalize(
        hist,
        hist,
        1.0,
        0.0,
        NORM_L1);

    for (int i = 0; i < histSize; i++)
    {
        feature.push_back(
            hist.at<float>(i));
    }

    return feature;
}

double MaterialMatcher::CosineSimilarity(
    const std::vector<float>& feature1,
    const std::vector<float>& feature2) const
{
    if (feature1.size() != feature2.size())
        return 0.0;

    if (feature1.empty())
        return 0.0;

    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;

    for (size_t i = 0; i < feature1.size(); i++)
    {
        dotProduct +=
            static_cast<double>(feature1[i]) *
            static_cast<double>(feature2[i]);

        norm1 +=
            static_cast<double>(feature1[i]) *
            static_cast<double>(feature1[i]);

        norm2 +=
            static_cast<double>(feature2[i]) *
            static_cast<double>(feature2[i]);
    }

    if (norm1 <= 0.0 || norm2 <= 0.0)
        return 0.0;

    return dotProduct / (std::sqrt(norm1) * std::sqrt(norm2));
}

//std::vector<float>
//MaterialMatcher::ExtractLBPFeature(
//    const cv::Mat& gray)
//{
//    std::vector<float> feature;
//
//
//    cv::Mat lbp =
//        cv::Mat::zeros(
//            gray.size(),
//            CV_8UC1);
//
//
//
//    for (int y = 1;
//        y < gray.rows - 1;
//        y++)
//    {
//
//        for (int x = 1;
//            x < gray.cols - 1;
//            x++)
//        {
//
//            uchar center =
//                gray.at<uchar>(y, x);
//
//
//            unsigned char code = 0;
//
//
//            code |=
//                (gray.at<uchar>(y - 1, x - 1) > center) << 7;
//
//            code |=
//                (gray.at<uchar>(y - 1, x) > center) << 6;
//
//            code |=
//                (gray.at<uchar>(y - 1, x + 1) > center) << 5;
//
//            code |=
//                (gray.at<uchar>(y, x + 1) > center) << 4;
//
//            code |=
//                (gray.at<uchar>(y + 1, x + 1) > center) << 3;
//
//            code |=
//                (gray.at<uchar>(y + 1, x) > center) << 2;
//
//            code |=
//                (gray.at<uchar>(y + 1, x - 1) > center) << 1;
//
//            code |=
//                (gray.at<uchar>(y, x - 1) > center);
//
//
//            lbp.at<uchar>(y, x) = code;
//        }
//    }
//
//
//
//    // LBP Histogram
//
//    int histSize = 256;
//
//    float range[] =
//    {
//        0,256
//    };
//
//    const float* histRange =
//    {
//        range
//    };
//
//
//    cv::Mat hist;
//
//
//    cv::calcHist(
//        &lbp,
//        1,
//        0,
//        cv::Mat(),
//        hist,
//        1,
//        &histSize,
//        &histRange);
//
//
//
//    cv::normalize(
//        hist,
//        hist,
//        1,
//        0,
//        cv::NORM_L1);
//
//
//
//    for (int i = 0; i < histSize; i++)
//    {
//        feature.push_back(
//            hist.at<float>(i));
//    }
//
//
//    return feature;
//}
