#pragma once

#include "material/FeatureExtractor.h"

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <vector>

namespace material
{

struct RecognizerOptions
{
    // 分類器訓練、拒判及各特徵群組權重設定。
    // Classifier training, rejection, and feature-group weight settings.
    FeatureOptions feature;

    bool enablePca = true;
    double pcaRetainedVariance = 0.98;
    int maxPcaComponents = 64;
    double covarianceShrinkage = 0.20;

    double rejectConfidence = 0.55;
    double rejectDistance = -1.0;

    std::map<std::string, double> featureWeights = {
        {"Color", 2.00},
        {"LBP", 1.40},
        {"Laws", 1.30},
        {"Gabor", 1.10},
        {"GLCM", 1.10},
        {"FFT", 1.00},
        {"Gradient", 0.85},
        {"LocalStd", 0.80}};
};

struct CandidateScore
{
    // 單一候選類別的排序分數及分群相似度。
    // Ranking scores and per-group similarity for one candidate class.
    std::string label;
    double confidence = 0.0;
    double similarity = 0.0;
    double distance = 0.0;
    std::map<std::string, double> featureSimilarity;
};

struct RecognitionResult
{
    // 最佳類別、是否通過拒判，以及依分數排序的全部候選。
    // Best class, rejection decision, and all candidates ordered by score.
    std::string label;
    bool accepted = false;
    double confidence = 0.0;
    double similarity = 0.0;
    double distance = 0.0;
    std::vector<CandidateScore> candidates;
};

class MaterialRecognizer
{
public:
    // 清除舊樣本後加入各類樣本，再呼叫 Train 建立模型。
    // Clear old samples, add labeled samples, then call Train to build a model.
    explicit MaterialRecognizer(RecognizerOptions options = {});

    void Clear();

    void AddMaterial(
        const std::string& label,
        const cv::Mat& image,
        const cv::Mat& roiMask = cv::Mat());

    void AddFeatureSample(
        const std::string& label,
        const std::vector<float>& feature);

    void Train();

    RecognitionResult Recognize(
        const cv::Mat& image,
        const cv::Mat& roiMask = cv::Mat()) const;

    RecognitionResult RecognizeFeature(
        const std::vector<float>& feature) const;

    void Save(const std::string& path) const;
    void Load(const std::string& path);

    bool IsTrained() const noexcept;
    std::size_t SampleCount() const noexcept;
    std::vector<std::string> Labels() const;
    const std::vector<FeatureRange>& FeatureLayout() const noexcept;
    const RecognizerOptions& options() const noexcept;

private:
    struct LabeledFeature
    {
        std::string label;
        std::vector<float> values;
    };

    RecognizerOptions options_;
    FeatureExtractor extractor_;
    std::vector<LabeledFeature> samples_;
    std::vector<FeatureRange> featureLayout_;

    bool trained_ = false;
    cv::Mat featureMean_;
    cv::Mat featureStd_;
    cv::Mat pcaMean_;
    cv::Mat pcaEigenvectors_;
    cv::Mat inverseCovariance_;

    std::map<std::string, cv::Mat> classCentroids_;
    std::map<std::string, cv::Mat> classStandardizedCentroids_;

    cv::Mat Standardize(
        const std::vector<float>& feature,
        bool applyWeights) const;

    cv::Mat Project(const cv::Mat& standardizedWeighted) const;
    double GroupWeight(const std::string& groupName) const;
    std::string BaseGroupName(const std::string& groupName) const;
    void ValidateOptions() const;
};

} // namespace material
