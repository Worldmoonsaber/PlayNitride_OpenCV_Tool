#pragma once

#include "material/MaterialRecognizer.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

struct LabeledMaterialRoi
{
    // 寫入模型資料庫的分類名稱，例如 "40"。
    // Class name written to the model database, for example "40".
    std::string label;

    // 產品標準影像中的固定樣本區域。
    // Fixed sample area in the product standard image.
    cv::Rect roi;
};

struct LabeledMaterialImage
{
    // 由父資料夾或設定指定的分類名稱。
    // Class label supplied by the parent folder or configuration.
    std::string label;

    // 已裁切完成、可直接作為訓練樣本的影像路徑。
    // Path to a pre-cropped image that can be used directly for training.
    std::filesystem::path imagePath;
};

// 集中保存產品專屬設定，避免散落在應用程式進入點。
// Keeps product-specific settings together instead of scattering them in main.
struct MaterialProductRecipe
{
    // 由外部系統直接指定完整模型檔案路徑，避免額外的產品 ID 命名規則。
    // The host supplies the complete model path, avoiding another product-ID rule.
    std::filesystem::path databasePath;

    // 修改 ROI、標籤或特徵設定後啟用一次；重建結果會供後續產線重複使用。
    // Enable once after changing ROIs, labels, or feature options; the rebuilt
    // database is reused by later production runs.
    bool forceRetrain = false;

    material::RecognizerOptions recognizerOptions;

    // 建議方式：每張檔案是一個完整樣本，不依賴標準圖的固定座標。
    // Preferred mode: each file is one complete sample with no fixed-grid dependency.
    std::vector<LabeledMaterialImage> trainingImages;

    // 相容舊方式：從單張標準圖依固定 ROI 取樣。
    // Legacy-compatible mode: crops fixed ROIs from one standard image.
    std::vector<LabeledMaterialRoi> trainingRois;
};

// 管理單一產品配方及其辨識器；Recognize 前必須先呼叫 Initialize。
// Owns one product recipe and recognizer; call Initialize before Recognize.
class MaterialProfile
{
public:
    explicit MaterialProfile(MaterialProductRecipe recipe);

    void Initialize();

    material::RecognitionResult Recognize(
        const cv::Mat& colorRoi,
        const cv::Mat& roiMask = cv::Mat()) const;

    // 透過支援 Unicode 的 Windows 路徑載入標準影像。
    // Loads the standard image through a Unicode-safe Windows path.

    bool IsInitialized() const noexcept;
    const MaterialProductRecipe& Recipe() const noexcept;
    const std::filesystem::path& DatabasePath() const noexcept;

private:
    // 將訓練與資料庫生命週期封裝起來，避免 main.cpp 管理內部細節。
    // Hides training and database lifecycle details from main.cpp.
    MaterialProductRecipe recipe_;
    material::MaterialRecognizer recognizer_;
    bool initialized_ = false;

    void TrainFromImageFiles();
    void ValidateRecipe() const;
    void ValidateTrainingRecipe() const;
    static cv::Mat LoadColorImage(const std::filesystem::path& path);
};
