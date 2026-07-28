#include "material/MaterialRecognizer.h"

#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace material
{
namespace
{

constexpr int kDatabaseVersion = 2;
constexpr double kEpsilon = 1.0e-12;

cv::Mat VectorToRow(
    const std::vector<float>& values,
    int depth = CV_64F)
{
    cv::Mat row(1, static_cast<int>(values.size()), CV_32F);
    if (!values.empty())
    {
        std::copy(values.begin(), values.end(), row.ptr<float>());
    }

    if (depth == CV_32F)
    {
        return row;
    }

    cv::Mat converted;
    row.convertTo(converted, depth);
    return converted;
}

std::vector<float> MatToVector(const cv::Mat& matrix)
{
    cv::Mat row = matrix.reshape(1, 1);
    cv::Mat row32f;
    row.convertTo(row32f, CV_32F);
    return std::vector<float>(
        row32f.ptr<float>(),
        row32f.ptr<float>() + row32f.cols);
}

bool SameLayout(
    const std::vector<FeatureRange>& left,
    const std::vector<FeatureRange>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }

    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (left[index].name != right[index].name ||
            left[index].begin != right[index].begin ||
            left[index].length != right[index].length)
        {
            return false;
        }
    }
    return true;
}

} // namespace

MaterialRecognizer::MaterialRecognizer(RecognizerOptions options)
    : options_(std::move(options)),
      extractor_(options_.feature)
{
    ValidateOptions();
}

void MaterialRecognizer::Clear()
{
    samples_.clear();
    featureLayout_.clear();
    classCentroids_.clear();
    classStandardizedCentroids_.clear();
    featureMean_.release();
    featureStd_.release();
    pcaMean_.release();
    pcaEigenvectors_.release();
    inverseCovariance_.release();
    trained_ = false;
}

void MaterialRecognizer::AddMaterial(
    const std::string& label,
    const cv::Mat& image,
    const cv::Mat& roiMask)
{
    if (label.empty())
    {
        throw std::invalid_argument("Material label cannot be empty.");
    }

    const ExtractionResult extracted =
        extractor_.Extract(image, roiMask);

    if (featureLayout_.empty())
    {
        featureLayout_ = extracted.groups;
    }
    else if (!SameLayout(featureLayout_, extracted.groups))
    {
        throw std::logic_error(
            "Feature layout changed between material samples.");
    }

    samples_.push_back({label, extracted.values});
    trained_ = false;
}

void MaterialRecognizer::AddFeatureSample(
    const std::string& label,
    const std::vector<float>& feature)
{
    if (label.empty())
    {
        throw std::invalid_argument("Material label cannot be empty.");
    }
    if (feature.empty())
    {
        throw std::invalid_argument("Feature vector cannot be empty.");
    }

    if (featureLayout_.empty())
    {
        featureLayout_.push_back(
            {"Custom@L0", 0, feature.size()});
    }

    const std::size_t expectedSize =
        featureLayout_.back().begin + featureLayout_.back().length;
    if (feature.size() != expectedSize)
    {
        throw std::invalid_argument(
            "Feature vector dimension does not match the database.");
    }

    samples_.push_back({label, feature});
    trained_ = false;
}

// 由所有已加入樣本建立標準化參數、PCA（選用）及各類中心點。
// Builds normalization, optional PCA, and class centroids from all samples.
void MaterialRecognizer::Train()
{
    if (samples_.empty())
    {
        throw std::logic_error("Add at least one material sample before Train.");
    }

    const int rowCount = static_cast<int>(samples_.size());
    const int dimension = static_cast<int>(samples_.front().values.size());
    if (dimension <= 0)
    {
        throw std::logic_error("Training feature vectors are empty.");
    }

    cv::Mat data(rowCount, dimension, CV_64F);
    for (int row = 0; row < rowCount; ++row)
    {
        if (static_cast<int>(samples_[row].values.size()) != dimension)
        {
            throw std::logic_error(
                "Training samples have inconsistent feature dimensions.");
        }
        for (int column = 0; column < dimension; ++column)
        {
            data.at<double>(row, column) =
                samples_[row].values[static_cast<std::size_t>(column)];
        }
    }

    cv::reduce(data, featureMean_, 0, cv::REDUCE_AVG, CV_64F);
    featureStd_ = cv::Mat::zeros(1, dimension, CV_64F);

    for (int column = 0; column < dimension; ++column)
    {
        double sumSquares = 0.0;
        for (int row = 0; row < rowCount; ++row)
        {
            const double difference =
                data.at<double>(row, column) -
                featureMean_.at<double>(0, column);
            sumSquares += difference * difference;
        }
        const double divisor = std::max(1, rowCount - 1);
        featureStd_.at<double>(0, column) =
            std::max(std::sqrt(sumSquares / divisor), 1.0e-6);
    }

    cv::Mat standardized(rowCount, dimension, CV_64F);
    for (int row = 0; row < rowCount; ++row)
    {
        for (int column = 0; column < dimension; ++column)
        {
            standardized.at<double>(row, column) =
                (data.at<double>(row, column) -
                 featureMean_.at<double>(0, column)) /
                featureStd_.at<double>(0, column);
        }
    }

    cv::Mat weighted = standardized.clone();
    for (const FeatureRange& group : featureLayout_)
    {
        const double scale = std::sqrt(GroupWeight(group.name));
        for (std::size_t offset = 0; offset < group.length; ++offset)
        {
            const int column =
                static_cast<int>(group.begin + offset);
            weighted.col(column) *= scale;
        }
    }

    pcaMean_.release();
    pcaEigenvectors_.release();
    if (options_.enablePca && rowCount > 1 && dimension > 1)
    {
        const int fullComponentCount =
            std::min(rowCount - 1, dimension);
        cv::PCA pca(
            weighted,
            cv::Mat(),
            cv::PCA::DATA_AS_ROW,
            fullComponentCount);

        const double totalVariance = cv::sum(pca.eigenvalues)[0];
        double retainedVariance = 0.0;
        int selectedComponents = 0;
        const int componentLimit = std::min(
            options_.maxPcaComponents,
            pca.eigenvectors.rows);

        while (selectedComponents < componentLimit)
        {
            retainedVariance +=
                pca.eigenvalues.at<double>(selectedComponents, 0);
            ++selectedComponents;
            if (totalVariance <= kEpsilon ||
                retainedVariance / totalVariance >=
                    options_.pcaRetainedVariance)
            {
                break;
            }
        }

        selectedComponents = std::max(selectedComponents, 1);
        pcaMean_ = pca.mean.clone();
        pcaEigenvectors_ =
            pca.eigenvectors.rowRange(0, selectedComponents).clone();
    }

    cv::Mat modelData;
    if (pcaEigenvectors_.empty())
    {
        modelData = weighted;
    }
    else
    {
        cv::Mat centered;
        cv::subtract(
            weighted,
            cv::repeat(pcaMean_, rowCount, 1),
            centered);
        modelData = centered * pcaEigenvectors_.t();
    }

    std::map<std::string, std::vector<int>> rowsByLabel;
    for (int row = 0; row < rowCount; ++row)
    {
        rowsByLabel[samples_[row].label].push_back(row);
    }

    classCentroids_.clear();
    classStandardizedCentroids_.clear();
    for (const auto& entry : rowsByLabel)
    {
        cv::Mat modelCentroid =
            cv::Mat::zeros(1, modelData.cols, CV_64F);
        cv::Mat standardizedCentroid =
            cv::Mat::zeros(1, standardized.cols, CV_64F);

        for (int row : entry.second)
        {
            modelCentroid += modelData.row(row);
            standardizedCentroid += standardized.row(row);
        }
        modelCentroid /= static_cast<double>(entry.second.size());
        standardizedCentroid /=
            static_cast<double>(entry.second.size());

        classCentroids_[entry.first] = modelCentroid;
        classStandardizedCentroids_[entry.first] =
            standardizedCentroid;
    }

    const int modelDimension = modelData.cols;
    cv::Mat covariance =
        cv::Mat::zeros(modelDimension, modelDimension, CV_64F);
    int withinClassDegreesOfFreedom = 0;

    for (const auto& entry : rowsByLabel)
    {
        const cv::Mat& centroid = classCentroids_.at(entry.first);
        for (int row : entry.second)
        {
            const cv::Mat difference = modelData.row(row) - centroid;
            covariance += difference.t() * difference;
        }
        withinClassDegreesOfFreedom +=
            std::max(0, static_cast<int>(entry.second.size()) - 1);
    }

    if (withinClassDegreesOfFreedom > 0)
    {
        covariance /= static_cast<double>(withinClassDegreesOfFreedom);
        double averageVariance =
            cv::trace(covariance)[0] /
            std::max(modelDimension, 1);
        averageVariance = std::max(averageVariance, 1.0e-6);

        const cv::Mat identity =
            cv::Mat::eye(modelDimension, modelDimension, CV_64F);
        covariance =
            (1.0 - options_.covarianceShrinkage) * covariance +
            options_.covarianceShrinkage *
                averageVariance * identity;
        covariance += 1.0e-6 * averageVariance * identity;
    }
    else
    {
        covariance =
            cv::Mat::eye(modelDimension, modelDimension, CV_64F);
    }

    if (!cv::invert(
            covariance,
            inverseCovariance_,
            cv::DECOMP_SVD))
    {
        throw std::runtime_error(
            "Could not invert the pooled feature covariance matrix.");
    }

    trained_ = true;
}

// 先擷取影像特徵，再交由共用的特徵向量分類流程處理。
// Extracts image features, then delegates to the shared vector classifier.
RecognitionResult MaterialRecognizer::Recognize(
    const cv::Mat& image,
    const cv::Mat& roiMask) const
{
    const ExtractionResult extracted =
        extractor_.Extract(image, roiMask);
    if (!SameLayout(featureLayout_, extracted.groups))
    {
        throw std::logic_error(
            "Input feature layout does not match the trained database.");
    }
    return RecognizeFeature(extracted.values);
}

// 計算各類距離、相似度與信心分數，最後套用拒判門檻。
// Computes class distance, similarity, confidence, then applies rejection limits.
RecognitionResult MaterialRecognizer::RecognizeFeature(
    const std::vector<float>& feature) const
{
    if (!trained_)
    {
        throw std::logic_error(
            "MaterialRecognizer must be trained or loaded before recognition.");
    }
    if (feature.size() != static_cast<std::size_t>(featureMean_.cols))
    {
        throw std::invalid_argument(
            "Recognition feature dimension does not match the database.");
    }

    const cv::Mat standardized =
        Standardize(feature, false);
    const cv::Mat weighted =
        Standardize(feature, true);
    const cv::Mat projected = Project(weighted);
    const double modelDimension =
        static_cast<double>(std::max(projected.cols, 1));

    RecognitionResult result;
    std::vector<double> logits;
    logits.reserve(classCentroids_.size());

    for (const auto& entry : classCentroids_)
    {
        CandidateScore candidate;
        candidate.label = entry.first;

        const cv::Mat difference = projected - entry.second;
        const cv::Mat distanceMatrix =
            difference * inverseCovariance_ * difference.t();
        const double distanceSquared =
            std::max(0.0, distanceMatrix.at<double>(0, 0));
        const double normalizedDistanceSquared =
            distanceSquared / modelDimension;

        candidate.distance =
            std::sqrt(normalizedDistanceSquared);
        candidate.similarity =
            std::exp(-0.5 * normalizedDistanceSquared);
        logits.push_back(-0.5 * normalizedDistanceSquared);

        const cv::Mat& classStandardized =
            classStandardizedCentroids_.at(entry.first);
        std::map<std::string, std::pair<double, std::size_t>>
            groupErrors;

        for (const FeatureRange& group : featureLayout_)
        {
            const std::string baseName =
                BaseGroupName(group.name);
            auto& accumulator = groupErrors[baseName];
            for (std::size_t offset = 0; offset < group.length; ++offset)
            {
                const int column =
                    static_cast<int>(group.begin + offset);
                const double value =
                    standardized.at<double>(0, column) -
                    classStandardized.at<double>(0, column);
                accumulator.first += value * value;
                accumulator.second += 1;
            }
        }

        for (const auto& groupError : groupErrors)
        {
            const double meanSquare =
                groupError.second.first /
                std::max<std::size_t>(
                    groupError.second.second, 1);
            candidate.featureSimilarity[groupError.first] =
                std::exp(-0.5 * meanSquare);
        }

        result.candidates.push_back(std::move(candidate));
    }

    const double maximumLogit =
        *std::max_element(logits.begin(), logits.end());
    double probabilitySum = 0.0;
    for (double& logit : logits)
    {
        logit = std::exp(logit - maximumLogit);
        probabilitySum += logit;
    }

    for (std::size_t index = 0;
         index < result.candidates.size();
         ++index)
    {
        result.candidates[index].confidence =
            logits[index] / std::max(probabilitySum, kEpsilon);
    }

    std::sort(
        result.candidates.begin(),
        result.candidates.end(),
        [](const CandidateScore& left, const CandidateScore& right)
        {
            return left.distance < right.distance;
        });

    const CandidateScore& best = result.candidates.front();
    result.label = best.label;
    result.confidence = best.confidence;
    result.similarity = best.similarity;
    result.distance = best.distance;
    result.accepted =
        result.confidence >= options_.rejectConfidence &&
        (options_.rejectDistance < 0.0 ||
         result.distance <= options_.rejectDistance);
    return result;
}

// 保存完整模型，讓產線啟動時無須重新訓練。
// Saves the complete model so production startup does not need retraining.
void MaterialRecognizer::Save(const std::string& path) const
{
    if (samples_.empty())
    {
        throw std::logic_error("Cannot save an empty material database.");
    }

    cv::FileStorage storage(
        path,
        cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!storage.isOpened())
    {
        throw std::runtime_error(
            "Could not open material database for writing: " + path);
    }

    storage << "databaseVersion" << kDatabaseVersion;
    storage << "options" << "{";
    storage << "enablePca" << static_cast<int>(options_.enablePca);
    storage << "pcaRetainedVariance" << options_.pcaRetainedVariance;
    storage << "maxPcaComponents" << options_.maxPcaComponents;
    storage << "covarianceShrinkage" << options_.covarianceShrinkage;
    storage << "rejectConfidence" << options_.rejectConfidence;
    storage << "rejectDistance" << options_.rejectDistance;

    const FeatureOptions& feature = options_.feature;
    storage << "feature" << "{";
    storage << "enableTextureFeatures"
            << static_cast<int>(feature.enableTextureFeatures);
    storage << "pyramidLevels" << feature.pyramidLevels;
    storage << "maxImageSide" << feature.maxImageSide;
    storage << "lowerPercentile" << feature.lowerPercentile;
    storage << "upperPercentile" << feature.upperPercentile;
    storage << "useClahe" << static_cast<int>(feature.useClahe);
    storage << "claheClipLimit" << feature.claheClipLimit;
    storage << "claheTileWidth" << feature.claheTileGrid.width;
    storage << "claheTileHeight" << feature.claheTileGrid.height;
    storage << "gaussianSigma" << feature.gaussianSigma;
    storage << "lbpPoints" << feature.lbpPoints;
    storage << "lbpRadius" << feature.lbpRadius;
    storage << "gradientBins" << feature.gradientBins;
    storage << "fftRings" << feature.fftRings;
    storage << "glcmLevels" << feature.glcmLevels;
    storage << "localStdWindow" << feature.localStdWindow;
    storage << "}";

    storage << "featureWeights" << "[";
    for (const auto& weight : options_.featureWeights)
    {
        storage << "{:"
                << "name" << weight.first
                << "value" << weight.second
                << "}";
    }
    storage << "]";
    storage << "}";

    storage << "featureLayout" << "[";
    for (const FeatureRange& group : featureLayout_)
    {
        storage << "{:"
                << "name" << group.name
                << "begin" << static_cast<int>(group.begin)
                << "length" << static_cast<int>(group.length)
                << "}";
    }
    storage << "]";

    storage << "samples" << "[";
    for (const LabeledFeature& sample : samples_)
    {
        storage << "{"
                << "label" << sample.label
                << "feature" << VectorToRow(sample.values, CV_32F)
                << "}";
    }
    storage << "]";
}

// 載入前驗證版本及必要欄位，避免使用不相容或不完整的模型。
// Validates version and required fields before accepting a persisted model.
void MaterialRecognizer::Load(const std::string& path)
{
    cv::FileStorage storage(path, cv::FileStorage::READ);
    if (!storage.isOpened())
    {
        throw std::runtime_error(
            "Could not open material database: " + path);
    }

    int version = 0;
    storage["databaseVersion"] >> version;
    if (version != kDatabaseVersion)
    {
        throw std::runtime_error(
            "Unsupported material database version.");
    }

    RecognizerOptions loadedOptions;
    const cv::FileNode optionsNode = storage["options"];
    int booleanValue = 0;
    optionsNode["enablePca"] >> booleanValue;
    loadedOptions.enablePca = booleanValue != 0;
    optionsNode["pcaRetainedVariance"] >>
        loadedOptions.pcaRetainedVariance;
    optionsNode["maxPcaComponents"] >>
        loadedOptions.maxPcaComponents;
    optionsNode["covarianceShrinkage"] >>
        loadedOptions.covarianceShrinkage;
    optionsNode["rejectConfidence"] >>
        loadedOptions.rejectConfidence;
    optionsNode["rejectDistance"] >>
        loadedOptions.rejectDistance;

    const cv::FileNode featureNode = optionsNode["feature"];
    FeatureOptions& feature = loadedOptions.feature;
    featureNode["enableTextureFeatures"] >> booleanValue;
    feature.enableTextureFeatures = booleanValue != 0;
    featureNode["pyramidLevels"] >> feature.pyramidLevels;
    featureNode["maxImageSide"] >> feature.maxImageSide;
    featureNode["lowerPercentile"] >> feature.lowerPercentile;
    featureNode["upperPercentile"] >> feature.upperPercentile;
    featureNode["useClahe"] >> booleanValue;
    feature.useClahe = booleanValue != 0;
    featureNode["claheClipLimit"] >> feature.claheClipLimit;
    featureNode["claheTileWidth"] >> feature.claheTileGrid.width;
    featureNode["claheTileHeight"] >> feature.claheTileGrid.height;
    featureNode["gaussianSigma"] >> feature.gaussianSigma;
    featureNode["lbpPoints"] >> feature.lbpPoints;
    featureNode["lbpRadius"] >> feature.lbpRadius;
    featureNode["gradientBins"] >> feature.gradientBins;
    featureNode["fftRings"] >> feature.fftRings;
    featureNode["glcmLevels"] >> feature.glcmLevels;
    featureNode["localStdWindow"] >> feature.localStdWindow;

    loadedOptions.featureWeights.clear();
    const cv::FileNode weightsNode = optionsNode["featureWeights"];
    for (const cv::FileNode& weightNode : weightsNode)
    {
        std::string name;
        double value = 1.0;
        weightNode["name"] >> name;
        weightNode["value"] >> value;
        loadedOptions.featureWeights[name] = value;
    }

    std::vector<FeatureRange> loadedLayout;
    const cv::FileNode layoutNode = storage["featureLayout"];
    for (const cv::FileNode& groupNode : layoutNode)
    {
        FeatureRange group;
        int begin = 0;
        int length = 0;
        groupNode["name"] >> group.name;
        groupNode["begin"] >> begin;
        groupNode["length"] >> length;
        group.begin = static_cast<std::size_t>(begin);
        group.length = static_cast<std::size_t>(length);
        loadedLayout.push_back(group);
    }

    std::vector<LabeledFeature> loadedSamples;
    const cv::FileNode samplesNode = storage["samples"];
    for (const cv::FileNode& sampleNode : samplesNode)
    {
        LabeledFeature sample;
        cv::Mat featureMatrix;
        sampleNode["label"] >> sample.label;
        sampleNode["feature"] >> featureMatrix;
        sample.values = MatToVector(featureMatrix);
        loadedSamples.push_back(std::move(sample));
    }

    if (loadedSamples.empty() || loadedLayout.empty())
    {
        throw std::runtime_error(
            "Material database contains no usable samples.");
    }

    options_ = std::move(loadedOptions);
    ValidateOptions();
    extractor_ = FeatureExtractor(options_.feature);
    samples_ = std::move(loadedSamples);
    featureLayout_ = std::move(loadedLayout);
    trained_ = false;
    Train();
}

bool MaterialRecognizer::IsTrained() const noexcept
{
    return trained_;
}

std::size_t MaterialRecognizer::SampleCount() const noexcept
{
    return samples_.size();
}

std::vector<std::string> MaterialRecognizer::Labels() const
{
    std::set<std::string> labels;
    for (const LabeledFeature& sample : samples_)
    {
        labels.insert(sample.label);
    }
    return {labels.begin(), labels.end()};
}

const std::vector<FeatureRange>&
MaterialRecognizer::FeatureLayout() const noexcept
{
    return featureLayout_;
}

const RecognizerOptions& MaterialRecognizer::options() const noexcept
{
    return options_;
}

cv::Mat MaterialRecognizer::Standardize(
    const std::vector<float>& feature,
    bool applyWeights) const
{
    cv::Mat standardized =
        (VectorToRow(feature) - featureMean_) / featureStd_;

    if (applyWeights)
    {
        for (const FeatureRange& group : featureLayout_)
        {
            const double scale =
                std::sqrt(GroupWeight(group.name));
            for (std::size_t offset = 0;
                 offset < group.length;
                 ++offset)
            {
                const int column =
                    static_cast<int>(group.begin + offset);
                standardized.at<double>(0, column) *= scale;
            }
        }
    }
    return standardized;
}

cv::Mat MaterialRecognizer::Project(
    const cv::Mat& standardizedWeighted) const
{
    if (pcaEigenvectors_.empty())
    {
        return standardizedWeighted.clone();
    }
    return (standardizedWeighted - pcaMean_) *
        pcaEigenvectors_.t();
}

double MaterialRecognizer::GroupWeight(
    const std::string& groupName) const
{
    const auto found =
        options_.featureWeights.find(BaseGroupName(groupName));
    return found == options_.featureWeights.end()
        ? 1.0
        : found->second;
}

std::string MaterialRecognizer::BaseGroupName(
    const std::string& groupName) const
{
    const std::size_t separator = groupName.find('@');
    return separator == std::string::npos
        ? groupName
        : groupName.substr(0, separator);
}

void MaterialRecognizer::ValidateOptions() const
{
    if (options_.pcaRetainedVariance <= 0.0 ||
        options_.pcaRetainedVariance > 1.0)
    {
        throw std::invalid_argument(
            "pcaRetainedVariance must be in (0, 1].");
    }
    if (options_.maxPcaComponents < 1)
    {
        throw std::invalid_argument(
            "maxPcaComponents must be positive.");
    }
    if (options_.covarianceShrinkage < 0.0 ||
        options_.covarianceShrinkage > 1.0)
    {
        throw std::invalid_argument(
            "covarianceShrinkage must be in [0, 1].");
    }
    if (options_.rejectConfidence < 0.0 ||
        options_.rejectConfidence > 1.0)
    {
        throw std::invalid_argument(
            "rejectConfidence must be in [0, 1].");
    }
    for (const auto& weight : options_.featureWeights)
    {
        if (weight.second <= 0.0)
        {
            throw std::invalid_argument(
                "All feature weights must be positive.");
        }
    }
}

} // namespace material
