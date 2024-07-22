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
    float Height();
    float Width();
    float Angle();
    vector<Point> Points();
    vector<Point> contour();
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
    float _Width = -1;
    float _Height = -1;
    float _Angle = -1;

};

vector<BlobInfo> RegionPartition(Mat& ImgBinary);
