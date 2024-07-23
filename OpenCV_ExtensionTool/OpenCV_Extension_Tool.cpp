
#include "OpenCV_Extension_Tool.h"

void RegionFloodFill(Mat& ImgBinary, int x, int y, vector<Point>& vectorPoint, vector<Point>& vContour)
{
	uchar tagIdx = 101;

	if (ImgBinary.at<uchar>(y, x) == 255)
	{
		ImgBinary.at<uchar>(y, x) = tagIdx;
		vectorPoint.push_back(Point(x, y));
	}
	else if (ImgBinary.at<uchar>(y, x) == tagIdx)
		return;

	int edgesSides = 0;

#pragma region 單一Region 較大時 容易出現資料堆積錯誤 專案屬性要調整

	for (int j = y - 1; j <= y + 1; j++)
		for (int i = x - 1; i <= x + 1; i++)
		{
			if (i == x && y == j)
				continue;


			if (i < 0 || j < 0)
			{
				edgesSides++;
				continue;
			}

			if (i >= ImgBinary.cols || j >= ImgBinary.rows)
			{
				edgesSides++;
				continue;
			}

			if (ImgBinary.at<uchar>(j, i) == 0)
			{
				edgesSides++;
				continue;
			}

			if (ImgBinary.at<uchar>(j, i) == 255)
				RegionFloodFill(ImgBinary, i, j, vectorPoint, vContour);

		}

#pragma endregion

	if (edgesSides > 1 && edgesSides < 8)
		vContour.push_back(Point(x, y));
}


vector<BlobInfo> RegionPartition(Mat& ImgBinary)
{
	vector<BlobInfo> lst;

	Mat ImgTag = ImgBinary.clone();

	for (int i = 0; i < ImgBinary.cols; i++)
		for (int j = 0; j < ImgBinary.rows; j++)
		{
			int val = ImgTag.at<uchar>(j, i);

			if (val == 255)
			{
				vector<Point> vArea;
				vector<Point> vContour;

				RegionFloodFill(ImgTag, i, j, vArea, vContour);

				BlobInfo regionInfo = BlobInfo(vArea, vContour);

				for (int i = 0; i < vArea.size(); i++)
					ImgTag.at<uchar>(vArea[i].y, vArea[i].x) = 0;

				lst.push_back(regionInfo);

			}

		}


	ImgTag.release();

	return lst;

}

BlobInfo::BlobInfo(vector<Point> vArea, vector<Point> vContour)
{
	_contour = vContour;
	_points = vArea;
	_area = vArea.size();


	float x_sum = 0, y_sum = 0;

	_XminBound = 99999;
	_YminBound = 99999;
	_XmaxBound = -1;
	_YmaxBound = -1;

	for (int i = 0; i < vArea.size(); i++)
	{
		x_sum += vArea[i].x;
		y_sum += vArea[i].y;

		if (vArea[i].x > _XmaxBound)
			_XmaxBound = vArea[i].x;

		if (vArea[i].y > _YmaxBound)
			_YmaxBound = vArea[i].y;

		if (vArea[i].x < _XminBound)
			_XminBound = vArea[i].x;

		if (vArea[i].y < _YminBound)
			_YminBound = vArea[i].y;

	}

	_center = Point2f(x_sum / vArea.size(), y_sum / vArea.size());

	float min_len = _area;
	float max_len = 0;

	for (int j = 0; j < vContour.size(); j++)
	{
		float xx = vContour[j].x - _center.x;
		float yy = vContour[j].y - _center.y;

		float len = sqrt((xx * xx + yy * yy));

		if (len > max_len)
			max_len = len;

		if (len < min_len)
			min_len = len;
	}

	_circularity = min_len / max_len;

	if (vContour.size() == 0)
		return;

	RotatedRect minrect = minAreaRect(vContour);

	_Angle=minrect.angle;

	while (_Angle < 0 && _Angle>180)
	{
		if (_Angle < 0)
			_Angle += 180;

		if (_Angle > 180)
			_Angle -= 180;
	}

	float minArea = (minrect.size.height + 1) * (minrect.size.width + 1);//擬合結果其實是內縮的 所以要+1

	_minRectHeight = minrect.size.height + 1;
	_minRectWidth = minrect.size.width + 1;

	if (_minRectHeight > _minRectWidth)
	{
		_AspectRatio = _minRectHeight / _minRectWidth;
		_Ra = _minRectHeight;
		_Rb = _minRectWidth;
	}
	else
	{
		_AspectRatio = _minRectWidth / _minRectHeight;
		_Rb = _minRectHeight;
		_Ra = _minRectWidth;

	}

	if (minArea > _area)
		_rectangularity = minArea / _area;
	else
		_rectangularity = _area / minArea;

	_rectangularity = abs(_rectangularity);

}

void BlobInfo::Release()
{
	_contour.clear();
	_points.clear();
}

int BlobInfo::Area()
{
	return _area;
}

float BlobInfo::Circularity()
{
	return _circularity;
}

Point2f BlobInfo::Center()
{
	return _center;
}

float BlobInfo::Rectangularity()
{
	return _rectangularity;
}

float BlobInfo::minRectHeight()
{
	return _minRectHeight;
}

float BlobInfo::minRectWidth()
{
	return _minRectWidth;
}

float BlobInfo::Angle()
{
	return _Angle;
}

float BlobInfo::AspectRatio()
{
	return _AspectRatio;
}

vector<Point> BlobInfo::Points()
{
	return _points;
}

vector<Point> BlobInfo::contour()
{
	return _contour;
}

/// <summary>
/// 長軸
/// </summary>
/// <returns></returns>
float BlobInfo::Ra()
{
	return _Ra;
}

/// <summary>
/// 短軸
/// </summary>
/// <returns></returns>
float BlobInfo::Rb()
{
	return _Rb;
}
