#pragma once

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp> //mophorlogical operation
#include<opencv2/core.hpp>
#include<vector>

using namespace cv;
using namespace std;

class BlobInfo
{
public:
    BlobInfo(vector<Point> vArea, vector<Point> vContour);
    BlobInfo();
    BlobInfo(Mat ImgRegion);
    BlobInfo(vector<Point> vContour);
    BlobInfo(vector<Point> vMainContour,vector<vector<Point>> vHollowContour);

    void CaculateBlob(vector<Point> vArea, vector<Point> vContour);
    void Release();
    int Area();

    /// <summary>
    ///  1 ~ 0.0   1:完美圓形
    /// </summary>
    /// <returns></returns>
    float Circularity();
    Point2f Center();
    /// <summary>
    ///  1 ~ 0.0   1:完美矩形
    /// </summary>
    /// <returns></returns>
    float Rectangularity();
    float minRectHeight();
    float minRectWidth();
    float Angle();
    /// <summary>
    /// 長寬比
    /// </summary>
    /// <returns></returns>
    float AspectRatio();
    vector<Point> Points();
    vector<Point> contour();

    /// <summary>
    /// 長軸長度
    /// </summary>
    /// <returns></returns>
    float Ra();

    /// <summary>
    /// 短軸長度
    /// </summary>
    /// <returns></returns>
    float Rb();

    int Xmin();
    int Ymin();
    int Xmax();
    int Ymax();
    int Width();
    int Height();
    /// <summary>
    /// 蓬鬆度
    /// </summary>
    /// <returns></returns>
    float Bulkiness();

    /// <summary>
    /// 緊緻度
    /// </summary>
    /// <returns></returns>
    float Compactness();

    /// <summary>
    /// 與 Circularity 有定義上的差別 這個屬性更適合偵測 中空圓環的圓環 圓環的 Roundness趨近於1 Circularity 0.1 左右
    /// </summary>
    /// <returns></returns>
    float Roundness();

    float Sides();


    /// <summary>
    /// Topology 才有用的屬性
    /// </summary>
    /// <returns></returns>
    vector<vector<Point>> contourHollow();

    /// <summary>
    /// Topology 才有用的屬性
    /// </summary>
    /// <returns></returns>
    vector<Point> contourMain();

private:

    int _area = -1;
    float _circularity = -1;
    float _rectangularity = -1;

    Point2f _center;

    vector<Point> _points;
    vector<Point> _contour;

    int _XminBound=-1;
    int _YminBound = -1;
    int _XmaxBound = -1;
    int _YmaxBound = -1;
    float _minRectWidth = -1;
    float _minRectHeight = -1;
    float _Angle = -1;
    float _AspectRatio = -1;
    float _Ra = -1;
    float _Rb = -1;
    float _bulkiness = -1;
    float _compactness = -1;
    float _roundness = -1;
    float _sides = -1;
    float _Width = -1;
    float _Height = -1;

    vector<Point> _contourMain;
    vector<vector<Point>> _contourHollow;

};

/// <summary>
/// 速度與記憶體使用量都在可接受範圍 建議使用這個方法
/// </summary>
/// <param name="ImgBinary"></param>
/// <returns></returns>
vector<BlobInfo> RegionPartitionTopology(Mat ImgBinary);
