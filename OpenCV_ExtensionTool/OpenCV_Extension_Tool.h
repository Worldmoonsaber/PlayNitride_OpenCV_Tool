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

void RegionFloodFill(Mat& ImgBinary, int x, int y, vector<Point>& vectorPoint, vector<Point>& vContour);

class BlobInfo
{
public:
    BlobInfo(vector<Point> vArea, vector<Point> vContour);

    void Release();
    int Area();
    float Circularity();
    Point2f Center();
    float Rectangularity();
    float minRectHeight();
    float minRectWidth();
    float Angle();
    float AspectRatio();
    vector<Point> Points();
    vector<Point> contour();
    float Ra();
    float Rb();
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
    /// <summary>
    /// ªø¼e¤ñ
    /// </summary>
    float _AspectRatio = -1;
    float _Ra = -1;
    float _Rb = -1;
};

vector<BlobInfo> RegionPartition(Mat& ImgBinary);
