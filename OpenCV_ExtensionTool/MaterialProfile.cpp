#include "MaterialProfile.h"

#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

MaterialProfile::MaterialProfile(MaterialProductRecipe recipe)
    : recipe_(std::move(recipe)),
      recognizer_(recipe_.recognizerOptions)
{
}

void MaterialProfile::Initialize()
{
    ValidateRecipe();
    const std::filesystem::path parentDirectory =
        recipe_.databasePath.parent_path();
    if (!parentDirectory.empty())
    {
        std::filesystem::create_directories(parentDirectory);
    }

    // 產線啟動時通常載入資料庫；僅在資料庫不存在或指定強制重訓時進行訓練。
    // Production startup normally loads the database; training occurs only
    // when it is missing or forceRetrain was explicitly requested.
    if (!recipe_.forceRetrain &&
        std::filesystem::exists(recipe_.databasePath))
    {
        recognizer_.Load(recipe_.databasePath.string());
    }
    else
    {
        ValidateTrainingRecipe();
        TrainFromImageFiles();
        recognizer_.Save(recipe_.databasePath.string());
    }

    initialized_ = true;
}

material::RecognitionResult MaterialProfile::Recognize(
    const cv::Mat& colorRoi,
    const cv::Mat& roiMask) const
{
    if (!initialized_)
    {
        throw std::logic_error(
            "MaterialProfile::Initialize must be called first.");
    }
    if (colorRoi.empty())
    {
        throw std::invalid_argument(
            "Cannot recognize an empty material ROI.");
    }
    return recognizer_.Recognize(colorRoi, roiMask);
}

cv::Mat MaterialProfile::LoadColorImage(
    const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error(
            "Cannot open material image: " + path.string());
    }

    const std::vector<unsigned char> encodedImage{
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()};
    const cv::Mat image = cv::imdecode(encodedImage, cv::IMREAD_COLOR);
    if (image.empty())
    {
        throw std::runtime_error(
            "Cannot decode material image: " + path.string());
    }
    return image;
}

bool MaterialProfile::IsInitialized() const noexcept
{
    return initialized_;
}

const MaterialProductRecipe&
MaterialProfile::Recipe() const noexcept
{
    return recipe_;
}

const std::filesystem::path&
MaterialProfile::DatabasePath() const noexcept
{
    return recipe_.databasePath;
}

void MaterialProfile::TrainFromImageFiles()
{
    // 每個檔案視為獨立樣本；資料夾名稱只負責提供分類標籤。
    // Treats every file as an independent sample; folder names provide labels.
    recognizer_.Clear();
    for (const LabeledMaterialImage& sample : recipe_.trainingImages)
    {
        const cv::Mat image = LoadColorImage(sample.imagePath);
        recognizer_.AddMaterial(sample.label, image);
    }
    recognizer_.Train();
}

void MaterialProfile::ValidateRecipe() const
{
    if (recipe_.databasePath.empty())
    {
        throw std::invalid_argument(
            "Material databasePath cannot be empty.");
    }
}

void MaterialProfile::ValidateTrainingRecipe() const
{
    if (recipe_.trainingImages.empty() && recipe_.trainingRois.empty())
    {
        throw std::invalid_argument(
            "Material recipe must contain training images or ROIs.");
    }
    for (const LabeledMaterialImage& sample : recipe_.trainingImages)
    {
        if (sample.label.empty() || sample.imagePath.empty())
        {
            throw std::invalid_argument(
                "Material recipe contains an invalid training image.");
        }
    }
    if (!recipe_.trainingRois.empty())
    {
        throw std::invalid_argument(
            "ROI-based training is not configured. Use pre-cropped "
            "trainingImages instead.");
    }
    for (const LabeledMaterialRoi& sample :
         recipe_.trainingRois)
    {
        if (sample.label.empty() ||
            sample.roi.width <= 0 ||
            sample.roi.height <= 0)
        {
            throw std::invalid_argument(
                "Material recipe contains an invalid training ROI.");
        }
    }
}
