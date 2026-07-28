#include "material/FeatureExtractor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace material
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

void ValidateMask(const cv::Mat& image, const cv::Mat& mask)
{
    if (!mask.empty() && mask.size() != image.size())
    {
        throw std::invalid_argument("ROI mask size must match the input image.");
    }
}

cv::Mat ToBinaryMask(const cv::Mat& source)
{
    if (source.empty())
    {
        return {};
    }

    cv::Mat gray;
    if (source.channels() == 1)
    {
        gray = source;
    }
    else if (source.channels() == 3)
    {
        cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    }
    else if (source.channels() == 4)
    {
        cv::cvtColor(source, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw std::invalid_argument("ROI mask must have 1, 3, or 4 channels.");
    }

    cv::Mat mask;
    cv::compare(gray, 0, mask, cv::CMP_GT);
    return mask;
}

std::vector<float> SamplePixels(
    const cv::Mat& image32f,
    const cv::Mat& mask,
    std::size_t maximumSamples = 1000000)
{
    const std::size_t total =
        static_cast<std::size_t>(image32f.rows) * image32f.cols;
    const int stride = std::max(
        1,
        static_cast<int>(std::sqrt(
            static_cast<double>(std::max<std::size_t>(total, 1)) /
            static_cast<double>(maximumSamples))));

    std::vector<float> values;
    values.reserve(std::min(total, maximumSamples));

    for (int y = 0; y < image32f.rows; y += stride)
    {
        const float* imageRow = image32f.ptr<float>(y);
        const uchar* maskRow = mask.empty() ? nullptr : mask.ptr<uchar>(y);

        for (int x = 0; x < image32f.cols; x += stride)
        {
            if (maskRow != nullptr && maskRow[x] == 0)
            {
                continue;
            }

            const float value = imageRow[x];
            if (std::isfinite(value))
            {
                values.push_back(value);
            }
        }
    }

    return values;
}

float QuantileInPlace(std::vector<float>& values, double quantile)
{
    if (values.empty())
    {
        throw std::invalid_argument("The ROI mask contains no valid pixels.");
    }

    quantile = std::clamp(quantile, 0.0, 1.0);
    const std::size_t index = static_cast<std::size_t>(
        std::llround(quantile * static_cast<double>(values.size() - 1)));
    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(index),
        values.end());
    return values[index];
}

std::pair<double, double> MeanAndStd(
    const cv::Mat& image,
    const cv::Mat& mask)
{
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(image, mean, stddev, mask);
    return {mean[0], stddev[0]};
}

double RootMeanSquare(const cv::Mat& image, const cv::Mat& mask)
{
    cv::Mat squared;
    cv::multiply(image, image, squared);
    return std::sqrt(std::max(0.0, cv::mean(squared, mask)[0]));
}

void NormalizeHistogram(std::vector<float>& histogram)
{
    const double sum = std::accumulate(
        histogram.begin(), histogram.end(), 0.0);
    if (sum > kEpsilon)
    {
        for (float& value : histogram)
        {
            value = static_cast<float>(value / sum);
        }
    }
}

float BilinearAt(const cv::Mat& image, float y, float x)
{
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.cols - 1);
    const int y1 = std::min(y0 + 1, image.rows - 1);
    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);

    const float top =
        (1.0F - wx) * image.at<uchar>(y0, x0) +
        wx * image.at<uchar>(y0, x1);
    const float bottom =
        (1.0F - wx) * image.at<uchar>(y1, x0) +
        wx * image.at<uchar>(y1, x1);
    return (1.0F - wy) * top + wy * bottom;
}

cv::Mat OuterProduct(
    const std::array<float, 5>& vertical,
    const std::array<float, 5>& horizontal)
{
    cv::Mat kernel(5, 5, CV_32F);
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            kernel.at<float>(y, x) = vertical[y] * horizontal[x];
        }
    }
    return kernel;
}

} // namespace

FeatureExtractor::FeatureExtractor(FeatureOptions options)
    : options_(std::move(options))
{
    if (options_.pyramidLevels < 1 || options_.pyramidLevels > 4)
    {
        throw std::invalid_argument("pyramidLevels must be between 1 and 4.");
    }
    if (options_.lowerPercentile < 0.0 ||
        options_.upperPercentile > 1.0 ||
        options_.lowerPercentile >= options_.upperPercentile)
    {
        throw std::invalid_argument(
            "Robust normalization percentiles are invalid.");
    }
    if (options_.lbpPoints < 8 || options_.lbpPoints > 24 ||
        options_.lbpRadius < 1)
    {
        throw std::invalid_argument(
            "LBP requires 8..24 points and a positive radius.");
    }
    if (options_.gradientBins < 4 || options_.fftRings < 2 ||
        options_.glcmLevels < 4 || options_.glcmLevels > 64)
    {
        throw std::invalid_argument("One or more feature dimensions are invalid.");
    }
    if (options_.localStdWindow < 3 ||
        options_.localStdWindow % 2 == 0)
    {
        throw std::invalid_argument(
            "localStdWindow must be an odd integer of at least 3.");
    }
}

const FeatureOptions& FeatureExtractor::options() const noexcept
{
    return options_;
}

ExtractionResult FeatureExtractor::Extract(
    const cv::Mat& image,
    const cv::Mat& roiMask) const
{
    if (image.empty())
    {
        throw std::invalid_argument("Cannot extract features from an empty image.");
    }
    ValidateMask(image, roiMask);

    ExtractionResult result;

    const auto append = [&result](
                            const std::string& name,
                            const std::vector<float>& values)
    {
        result.groups.push_back(
            {name, result.values.size(), values.size()});
        result.values.insert(
            result.values.end(), values.begin(), values.end());
    };

    cv::Mat mask = ToBinaryMask(roiMask);
    append("Color", ExtractColor(image, mask));
    if (!options_.enableTextureFeatures)
    {
        return result;
    }

    cv::Mat levelImage = PrepareImage(image, mask);
    const int minimumSide =
        2 * options_.lbpRadius + 8;
    const int requiredSide =
        minimumSide * (1 << (options_.pyramidLevels - 1));
    if (std::min(levelImage.rows, levelImage.cols) < requiredSide)
    {
        throw std::invalid_argument(
            "Input image is too small for the configured pyramid.");
    }

    for (int level = 0; level < options_.pyramidLevels; ++level)
    {
        const std::string suffix = "@L" + std::to_string(level);
        append("LBP" + suffix, ExtractLbp(levelImage, mask));
        append("Gabor" + suffix, ExtractGabor(levelImage, mask));
        append("Gradient" + suffix, ExtractGradient(levelImage, mask));
        append("FFT" + suffix, ExtractFftRings(levelImage, mask));
        append("GLCM" + suffix, ExtractGlcm(levelImage, mask));
        append("LocalStd" + suffix, ExtractLocalStd(levelImage, mask));
        append("Laws" + suffix, ExtractLaws(levelImage, mask));

        if (level + 1 < options_.pyramidLevels)
        {
            cv::Mat nextImage;
            cv::pyrDown(levelImage, nextImage);
            levelImage = nextImage;

            if (!mask.empty())
            {
                cv::Mat nextMask;
                cv::resize(
                    mask,
                    nextMask,
                    levelImage.size(),
                    0.0,
                    0.0,
                    cv::INTER_NEAREST);
                mask = nextMask;
            }
        }
    }

    return result;
}

cv::Mat FeatureExtractor::PrepareImage(
    const cv::Mat& image,
    cv::Mat& mask) const
{
    cv::Mat gray;
    if (image.channels() == 1)
    {
        gray = image;
    }
    else if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw std::invalid_argument(
            "Input image must have 1, 3, or 4 channels.");
    }

    if (options_.maxImageSide > 0)
    {
        const int largestSide = std::max(gray.rows, gray.cols);
        if (largestSide > options_.maxImageSide)
        {
            const double scale =
                static_cast<double>(options_.maxImageSide) / largestSide;
            cv::resize(
                gray,
                gray,
                cv::Size(),
                scale,
                scale,
                cv::INTER_AREA);

            if (!mask.empty())
            {
                cv::resize(
                    mask,
                    mask,
                    gray.size(),
                    0.0,
                    0.0,
                    cv::INTER_NEAREST);
            }
        }
    }

    cv::Mat gray32f;
    gray.convertTo(gray32f, CV_32F);
    std::vector<float> samples = SamplePixels(gray32f, mask);
    const float low = QuantileInPlace(samples, options_.lowerPercentile);
    const float high = QuantileInPlace(samples, options_.upperPercentile);

    cv::Mat normalized32f;
    if (high - low <= std::numeric_limits<float>::epsilon())
    {
        normalized32f = cv::Mat::zeros(gray.size(), CV_32F);
    }
    else
    {
        normalized32f = (gray32f - low) * (255.0F / (high - low));
        cv::max(normalized32f, 0.0, normalized32f);
        cv::min(normalized32f, 255.0, normalized32f);
    }

    cv::Mat normalized;
    normalized32f.convertTo(normalized, CV_8U);

    if (options_.useClahe)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
            options_.claheClipLimit,
            options_.claheTileGrid);
        clahe->apply(normalized, normalized);
    }

    if (options_.gaussianSigma > 0.0)
    {
        cv::GaussianBlur(
            normalized,
            normalized,
            cv::Size(),
            options_.gaussianSigma,
            options_.gaussianSigma,
            cv::BORDER_REFLECT101);
    }

    return normalized;
}

std::vector<float> FeatureExtractor::ExtractColor(
    const cv::Mat& image,
    const cv::Mat& mask) const
{
    cv::Mat color;
    if (image.channels() == 1)
    {
        cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    }
    else if (image.channels() == 3)
    {
        color = image;
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, color, cv::COLOR_BGRA2BGR);
    }
    else
    {
        throw std::invalid_argument(
            "Input image must have 1, 3, or 4 channels.");
    }

    double inputScale = 1.0;
    if (color.depth() == CV_16U)
    {
        inputScale = 255.0 / 65535.0;
    }
    else if (color.depth() == CV_32F ||
             color.depth() == CV_64F)
    {
        double maximum = 0.0;
        cv::minMaxLoc(color.reshape(1), nullptr, &maximum);
        if (maximum <= 1.5)
        {
            inputScale = 255.0;
        }
    }

    cv::Mat color32f;
    color.convertTo(color32f, CV_32F, inputScale);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(color32f, mean, stddev, mask);

    const double blue = mean[0];
    const double green = std::max(mean[1], 1.0);
    const double red = mean[2];

    return {
        static_cast<float>(red / green),
        static_cast<float>(blue / green),
        static_cast<float>((mean[1] - red) / 255.0),
        static_cast<float>(stddev[0] / 255.0)};
}

std::vector<float> FeatureExtractor::ExtractLbp(
    const cv::Mat& image,
    const cv::Mat& mask) const
{
    const int points = options_.lbpPoints;
    const int radius = options_.lbpRadius;
    std::vector<float> histogram(
        static_cast<std::size_t>(points + 2), 0.0F);
    std::vector<uchar> bits(static_cast<std::size_t>(points));

    for (int y = radius; y < image.rows - radius; ++y)
    {
        const uchar* maskRow = mask.empty() ? nullptr : mask.ptr<uchar>(y);
        for (int x = radius; x < image.cols - radius; ++x)
        {
            if (maskRow != nullptr && maskRow[x] == 0)
            {
                continue;
            }

            const float center = image.at<uchar>(y, x);
            int ones = 0;
            for (int p = 0; p < points; ++p)
            {
                const double angle =
                    2.0 * CV_PI * static_cast<double>(p) / points;
                const float sampleX =
                    static_cast<float>(x + radius * std::cos(angle));
                const float sampleY =
                    static_cast<float>(y - radius * std::sin(angle));
                bits[static_cast<std::size_t>(p)] =
                    BilinearAt(image, sampleY, sampleX) >= center ? 1 : 0;
                ones += bits[static_cast<std::size_t>(p)];
            }

            int transitions = 0;
            for (int p = 0; p < points; ++p)
            {
                transitions +=
                    bits[static_cast<std::size_t>(p)] !=
                    bits[static_cast<std::size_t>((p + 1) % points)];
            }

            const int bin = transitions <= 2 ? ones : points + 1;
            histogram[static_cast<std::size_t>(bin)] += 1.0F;
        }
    }

    NormalizeHistogram(histogram);
    return histogram;
}

std::vector<float> FeatureExtractor::ExtractGabor(
    const cv::Mat& image,
    const cv::Mat& mask) const
{
    cv::Mat image32f;
    image.convertTo(image32f, CV_32F);
    const auto [imageMean, imageStd] = MeanAndStd(image32f, mask);
    image32f -= imageMean;
    const double scale = std::max(imageStd, 1.0);

    const std::array<double, 4> angles = {
        0.0, CV_PI / 4.0, CV_PI / 2.0, 3.0 * CV_PI / 4.0};
    const std::array<double, 3> wavelengths = {4.0, 8.0, 16.0};

    std::vector<float> feature;
    feature.reserve(angles.size() * wavelengths.size());

    for (double wavelength : wavelengths)
    {
        for (double angle : angles)
        {
            cv::Mat kernel = cv::getGaborKernel(
                cv::Size(31, 31),
                0.55 * wavelength,
                angle,
                wavelength,
                0.5,
                0.0,
                CV_32F);
            kernel -= cv::mean(kernel)[0];
            const double norm = cv::norm(kernel, cv::NORM_L2);
            if (norm > kEpsilon)
            {
                kernel /= norm;
            }

            cv::Mat response;
            cv::filter2D(
                image32f,
                response,
                CV_32F,
                kernel,
                cv::Point(-1, -1),
                0.0,
                cv::BORDER_REFLECT101);
            feature.push_back(
                static_cast<float>(RootMeanSquare(response, mask) / scale));
        }
    }

    return feature;
}

std::vector<float> FeatureExtractor::ExtractGradient(
    const cv::Mat& image,
    const cv::Mat& mask) const
{
    cv::Mat image32f;
    image.convertTo(image32f, CV_32F);

    cv::Mat gradientX;
    cv::Mat gradientY;
    cv::Scharr(image32f, gradientX, CV_32F, 1, 0);
    cv::Scharr(image32f, gradientY, CV_32F, 0, 1);

    cv::Mat magnitude;
    cv::Mat angle;
    cv::cartToPolar(gradientX, gradientY, magnitude, angle, true);

    std::vector<float> histogram(
        static_cast<std::size_t>(options_.gradientBins), 0.0F);
    double totalMagnitude = 0.0;

    for (int y = 0; y < image.rows; ++y)
    {
        const float* magnitudeRow = magnitude.ptr<float>(y);
        const float* angleRow = angle.ptr<float>(y);
        const uchar* maskRow = mask.empty() ? nullptr : mask.ptr<uchar>(y);

        for (int x = 0; x < image.cols; ++x)
        {
            if (maskRow != nullptr && maskRow[x] == 0)
            {
                continue;
            }

            const float unsignedAngle = std::fmod(angleRow[x], 180.0F);
            const int bin = std::min(
                options_.gradientBins - 1,
                static_cast<int>(
                    unsignedAngle * options_.gradientBins / 180.0F));
            histogram[static_cast<std::size_t>(bin)] += magnitudeRow[x];
            totalMagnitude += magnitudeRow[x];
        }
    }

    if (totalMagnitude > kEpsilon)
    {
        for (float& value : histogram)
        {
            value = static_cast<float>(value / totalMagnitude);
        }
    }

    const auto [imageMean, imageStd] = MeanAndStd(image32f, mask);
    const auto [magnitudeMean, magnitudeStd] = MeanAndStd(magnitude, mask);
    const double denominator = std::max(imageStd, 1.0);
    histogram.push_back(
        static_cast<float>(magnitudeMean / denominator));
    histogram.push_back(
        static_cast<float>(magnitudeStd / denominator));
    return histogram;
}

std::vector<float> FeatureExtractor::ExtractFftRings(
    const cv::Mat& image,
    const cv::Mat& mask) const
{
    cv::Mat image32f;
    image.convertTo(image32f, CV_32F);
    const auto [mean, stddev] = MeanAndStd(image32f, mask);

    if (!mask.empty())
    {
        image32f.setTo(mean, mask == 0);
    }
    image32f -= mean;

    cv::Mat window;
    cv::createHanningWindow(window, image.size(), CV_32F);
    image32f = image32f.mul(window);

    const int rows = cv::getOptimalDFTSize(image.rows);
    const int cols = cv::getOptimalDFTSize(image.cols);
    cv::Mat padded;
    cv::copyMakeBorder(
        image32f,
        padded,
        0,
        rows - image.rows,
        0,
        cols - image.cols,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0));

    cv::Mat spectrum;
    cv::dft(padded, spectrum, cv::DFT_COMPLEX_OUTPUT);
    std::vector<cv::Mat> planes;
    cv::split(spectrum, planes);

    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    magnitude += 1.0F;
    cv::log(magnitude, magnitude);

    std::vector<double> sums(
        static_cast<std::size_t>(options_.fftRings), 0.0);
    std::vector<std::size_t> counts(
        static_cast<std::size_t>(options_.fftRings), 0);

    for (int y = 0; y < magnitude.rows; ++y)
    {
        const float* row = magnitude.ptr<float>(y);
        const double fy =
            static_cast<double>(std::min(y, magnitude.rows - y)) /
            std::max(1.0, magnitude.rows / 2.0);

        for (int x = 0; x < magnitude.cols; ++x)
        {
            if (x == 0 && y == 0)
            {
                continue;
            }

            const double fx =
                static_cast<double>(std::min(x, magnitude.cols - x)) /
                std::max(1.0, magnitude.cols / 2.0);
            const double radius = std::sqrt(fx * fx + fy * fy);
            if (radius >= 1.0)
            {
                continue;
            }

            const int ring = std::min(
                options_.fftRings - 1,
                static_cast<int>(radius * options_.fftRings));
            sums[static_cast<std::size_t>(ring)] += row[x];
            counts[static_cast<std::size_t>(ring)] += 1;
        }
    }

    std::vector<float> feature(
        static_cast<std::size_t>(options_.fftRings), 0.0F);
    for (int ring = 0; ring < options_.fftRings; ++ring)
    {
        if (counts[static_cast<std::size_t>(ring)] > 0)
        {
            feature[static_cast<std::size_t>(ring)] =
                static_cast<float>(
                    sums[static_cast<std::size_t>(ring)] /
                    counts[static_cast<std::size_t>(ring)]);
        }
    }
    NormalizeHistogram(feature);

    if (stddev <= kEpsilon)
    {
        std::fill(feature.begin(), feature.end(), 0.0F);
    }
    return feature;
}

std::vector<float> FeatureExtractor::ExtractGlcm(
    const cv::Mat& image,
    const cv::Mat& mask) const
{
    const int levels = options_.glcmLevels;
    cv::Mat quantized;
    image.convertTo(
        quantized,
        CV_8U,
        static_cast<double>(levels) / 256.0);
    cv::min(quantized, levels - 1, quantized);

    const std::array<cv::Point, 4> offsets = {
        cv::Point(1, 0),
        cv::Point(1, 1),
        cv::Point(0, 1),
        cv::Point(-1, 1)};

    std::array<double, 6> totals = {};
    int validDirections = 0;

    for (const cv::Point& offset : offsets)
    {
        std::vector<double> matrix(
            static_cast<std::size_t>(levels * levels), 0.0);
        double pairCount = 0.0;

        for (int y = 0; y < image.rows; ++y)
        {
            const int neighborY = y + offset.y;
            if (neighborY < 0 || neighborY >= image.rows)
            {
                continue;
            }

            for (int x = 0; x < image.cols; ++x)
            {
                const int neighborX = x + offset.x;
                if (neighborX < 0 || neighborX >= image.cols)
                {
                    continue;
                }
                if (!mask.empty() &&
                    (mask.at<uchar>(y, x) == 0 ||
                     mask.at<uchar>(neighborY, neighborX) == 0))
                {
                    continue;
                }

                const int a = quantized.at<uchar>(y, x);
                const int b = quantized.at<uchar>(neighborY, neighborX);
                matrix[static_cast<std::size_t>(a * levels + b)] += 1.0;
                matrix[static_cast<std::size_t>(b * levels + a)] += 1.0;
                pairCount += 2.0;
            }
        }

        if (pairCount <= 0.0)
        {
            continue;
        }
        ++validDirections;
        for (double& value : matrix)
        {
            value /= pairCount;
        }

        double meanI = 0.0;
        double meanJ = 0.0;
        for (int i = 0; i < levels; ++i)
        {
            for (int j = 0; j < levels; ++j)
            {
                const double probability =
                    matrix[static_cast<std::size_t>(i * levels + j)];
                meanI += i * probability;
                meanJ += j * probability;
            }
        }

        double varianceI = 0.0;
        double varianceJ = 0.0;
        double contrast = 0.0;
        double homogeneity = 0.0;
        double asmValue = 0.0;
        double entropy = 0.0;
        double covariance = 0.0;

        for (int i = 0; i < levels; ++i)
        {
            for (int j = 0; j < levels; ++j)
            {
                const double probability =
                    matrix[static_cast<std::size_t>(i * levels + j)];
                if (probability <= 0.0)
                {
                    continue;
                }

                const double difference = static_cast<double>(i - j);
                contrast += difference * difference * probability;
                homogeneity +=
                    probability / (1.0 + std::abs(difference));
                asmValue += probability * probability;
                entropy -= probability * std::log(probability);
                varianceI +=
                    (i - meanI) * (i - meanI) * probability;
                varianceJ +=
                    (j - meanJ) * (j - meanJ) * probability;
                covariance +=
                    (i - meanI) * (j - meanJ) * probability;
            }
        }

        const double correlation =
            covariance /
            std::sqrt(std::max(
                varianceI * varianceJ,
                kEpsilon));
        const double normalization =
            static_cast<double>((levels - 1) * (levels - 1));

        totals[0] += contrast / std::max(normalization, 1.0);
        totals[1] += homogeneity;
        totals[2] += asmValue;
        totals[3] += std::sqrt(asmValue);
        totals[4] += entropy / std::log(static_cast<double>(levels * levels));
        totals[5] += correlation;
    }

    std::vector<float> feature(6, 0.0F);
    if (validDirections > 0)
    {
        for (std::size_t index = 0; index < feature.size(); ++index)
        {
            feature[index] = static_cast<float>(
                totals[index] / validDirections);
        }
    }
    return feature;
}

std::vector<float> FeatureExtractor::ExtractLocalStd(
    const cv::Mat& image,
    const cv::Mat& mask) const
{
    cv::Mat image32f;
    image.convertTo(image32f, CV_32F);

    cv::Mat mean;
    cv::Mat meanSquare;
    cv::boxFilter(
        image32f,
        mean,
        CV_32F,
        cv::Size(options_.localStdWindow, options_.localStdWindow),
        cv::Point(-1, -1),
        true,
        cv::BORDER_REFLECT101);
    cv::boxFilter(
        image32f.mul(image32f),
        meanSquare,
        CV_32F,
        cv::Size(options_.localStdWindow, options_.localStdWindow),
        cv::Point(-1, -1),
        true,
        cv::BORDER_REFLECT101);

    cv::Mat variance = meanSquare - mean.mul(mean);
    cv::max(variance, 0.0, variance);
    cv::Mat localStd;
    cv::sqrt(variance, localStd);

    const auto [imageMean, imageStd] = MeanAndStd(image32f, mask);
    const auto [localMean, localStddev] = MeanAndStd(localStd, mask);
    const double denominator = std::max(imageStd, 1.0);

    std::vector<float> pixels = SamplePixels(localStd, mask);
    const float percentile75 = QuantileInPlace(pixels, 0.75);
    const float percentile90 = QuantileInPlace(pixels, 0.90);

    return {
        static_cast<float>(localMean / denominator),
        static_cast<float>(localStddev / denominator),
        static_cast<float>(percentile75 / denominator),
        static_cast<float>(percentile90 / denominator)};
}

std::vector<float> FeatureExtractor::ExtractLaws(
    const cv::Mat& image,
    const cv::Mat& mask) const
{
    static const std::array<std::array<float, 5>, 5> vectors = {{
        {{1.0F, 4.0F, 6.0F, 4.0F, 1.0F}},
        {{-1.0F, -2.0F, 0.0F, 2.0F, 1.0F}},
        {{-1.0F, 0.0F, 2.0F, 0.0F, -1.0F}},
        {{-1.0F, 2.0F, 0.0F, -2.0F, 1.0F}},
        {{1.0F, -4.0F, 6.0F, -4.0F, 1.0F}}
    }};

    cv::Mat image32f;
    image.convertTo(image32f, CV_32F);
    const auto [mean, stddev] = MeanAndStd(image32f, mask);
    image32f -= mean;
    const double denominator = std::max(stddev, 1.0);

    std::vector<float> feature;
    feature.reserve(14);

    for (int first = 0; first < 5; ++first)
    {
        for (int second = first; second < 5; ++second)
        {
            if (first == 0 && second == 0)
            {
                continue;
            }

            cv::Mat response;
            cv::filter2D(
                image32f,
                response,
                CV_32F,
                OuterProduct(vectors[first], vectors[second]),
                cv::Point(-1, -1),
                0.0,
                cv::BORDER_REFLECT101);
            double energy = RootMeanSquare(response, mask);

            if (first != second)
            {
                cv::Mat transposedResponse;
                cv::filter2D(
                    image32f,
                    transposedResponse,
                    CV_32F,
                    OuterProduct(vectors[second], vectors[first]),
                    cv::Point(-1, -1),
                    0.0,
                    cv::BORDER_REFLECT101);
                energy =
                    0.5 * (energy + RootMeanSquare(transposedResponse, mask));
            }

            feature.push_back(
                static_cast<float>(energy / denominator));
        }
    }

    return feature;
}

} // namespace material
