#pragma once
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct MaterialFeature
{
    int LightIntensity;                  // 材質名稱
    std::vector<float> GrayFeature;        // 特徵向量
    // LBP紋理
    std::vector<float> LBPFeature;
    // Gabor方向紋理
    std::vector<float> GaborFeature;

    int SampleCount = 0;               // 樣本數
};

struct MaterialMatchResult
{
    bool Found = false;

    int Index = -1;

    std::string Name;

    double Score = 0.0;


    std::unordered_map<int, double> dicResultScores;

};

class MaterialMatcher
{
public:

    MaterialMatcher() = default;
    ~MaterialMatcher() = default;

    //--------------------------------------------------
    // 建立一種材質並加入資料庫
    //--------------------------------------------------
    bool BuildMaterial(
        int nLightIntensity,
        const std::vector<cv::Mat>& sampleImages);

    //--------------------------------------------------
    // 手動加入
    //--------------------------------------------------
    void AddMaterial(
        const MaterialFeature& material);

    //--------------------------------------------------
    // 清空資料庫
    //--------------------------------------------------
    void ClearDatabase();

    //--------------------------------------------------
    // 取得資料庫
    //--------------------------------------------------
    const std::vector<MaterialFeature>&
        GetDatabase() const;

    MaterialMatchResult Match(
        const cv::Mat& image);

private:

    //--------------------------------------------------
    // 特徵抽取
    //--------------------------------------------------
    std::vector<float> ExtractFeature(
        const cv::Mat& image);

    std::vector<float>
        ExtractGrayFeature(
            const cv::Mat& gray);

    std::vector<float>
        ExtractLBPFeature(
            const cv::Mat& gray);

    std::vector<float>
        ExtractGaborFeature(
            const cv::Mat& gray);


    double CosineSimilarity(
        const std::vector<float>& feature1,
        const std::vector<float>& feature2) const;
private:

    std::vector<MaterialFeature> m_Database;
};