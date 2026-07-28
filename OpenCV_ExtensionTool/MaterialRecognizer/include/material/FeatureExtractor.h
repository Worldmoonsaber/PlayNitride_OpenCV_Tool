#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace material
{

struct FeatureOptions
{
    bool enableTextureFeatures = true;
    int pyramidLevels = 2;
    int maxImageSide = 1024;

    double lowerPercentile = 0.01;
    double upperPercentile = 0.99;
    bool useClahe = true;
    double claheClipLimit = 2.0;
    cv::Size claheTileGrid = cv::Size(8, 8);
    double gaussianSigma = 1.0;

    int lbpPoints = 16;
    int lbpRadius = 2;
    int gradientBins = 8;
    int fftRings = 8;
    int glcmLevels = 16;
    int localStdWindow = 11;
};

struct FeatureRange
{
    std::string name;
    std::size_t begin = 0;
    std::size_t length = 0;
};

struct ExtractionResult
{
    std::vector<float> values;
    std::vector<FeatureRange> groups;
};

class FeatureExtractor
{
public:
    explicit FeatureExtractor(FeatureOptions options = {});

    const FeatureOptions& options() const noexcept;

    ExtractionResult Extract(
        const cv::Mat& image,
        const cv::Mat& roiMask = cv::Mat()) const;

private:
    FeatureOptions options_;

    cv::Mat PrepareImage(const cv::Mat& image, cv::Mat& mask) const;

    std::vector<float> ExtractColor(
        const cv::Mat& image,
        const cv::Mat& mask) const;

    std::vector<float> ExtractLbp(
        const cv::Mat& image,
        const cv::Mat& mask) const;

    std::vector<float> ExtractGabor(
        const cv::Mat& image,
        const cv::Mat& mask) const;

    std::vector<float> ExtractGradient(
        const cv::Mat& image,
        const cv::Mat& mask) const;

    std::vector<float> ExtractFftRings(
        const cv::Mat& image,
        const cv::Mat& mask) const;

    std::vector<float> ExtractGlcm(
        const cv::Mat& image,
        const cv::Mat& mask) const;

    std::vector<float> ExtractLocalStd(
        const cv::Mat& image,
        const cv::Mat& mask) const;

    std::vector<float> ExtractLaws(
        const cv::Mat& image,
        const cv::Mat& mask) const;
};

} // namespace material
