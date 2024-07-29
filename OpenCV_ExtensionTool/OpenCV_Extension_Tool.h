#pragma once

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp> //mophorlogical operation
#include<opencv2/core.hpp>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

void RegionFloodFill(Mat& ImgBinary, int x, int y, vector<Point>& vectorPoint, vector<Point>& vContour,int maxArea, bool& isOverSizeExtension);

class BlobInfo
{
public:
    BlobInfo(vector<Point> vArea, vector<Point> vContour);

    void Release();
    int Area();

    /// <summary>
    ///  1 ~ 0.0   1:�������
    /// </summary>
    /// <returns></returns>
    float Circularity();
    Point2f Center();
    /// <summary>
    ///  1 ~ 0.0   1:�����x��
    /// </summary>
    /// <returns></returns>
    float Rectangularity();
    float minRectHeight();
    float minRectWidth();
    float Angle();
    /// <summary>
    /// ���e��
    /// </summary>
    /// <returns></returns>
    float AspectRatio();
    vector<Point> Points();
    vector<Point> contour();

    /// <summary>
    /// ���b����
    /// </summary>
    /// <returns></returns>
    float Ra();

    /// <summary>
    /// �u�b����
    /// </summary>
    /// <returns></returns>
    float Rb();

    int Xmin();
    int Ymin();
    int Xmax();
    int Ymax();

    /// <summary>
    /// ���P��
    /// </summary>
    /// <returns></returns>
    float Bulkiness();

    /// <summary>
    /// ��o��
    /// </summary>
    /// <returns></returns>
    float Compactness();
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
};

/// <summary>
/// 
/// </summary>
/// <param name="ImgBinary"></param>
/// <param name="maxArea">�O�@���I �p�G���ݭn�o��j��Region �i�H�b�o�����α����o�� �קK�O�����n���D����</param>
/// <returns></returns>
vector<BlobInfo> RegionPartition(Mat ImgBinary,int maxArea= INT_MAX-2,int minArea=-1);
