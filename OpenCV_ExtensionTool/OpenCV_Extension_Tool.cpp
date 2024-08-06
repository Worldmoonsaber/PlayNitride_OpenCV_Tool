
#include "OpenCV_Extension_Tool.h"

/// <summary>
/// 
/// </summary>
/// <param name="ImgBinary"></param>
/// <param name="x"></param>
/// <param name="y"></param>
/// <param name="vectorPoint"></param>
/// <param name="vContour"></param>
/// <param name="maxArea"></param>
/// <param name="isOverSizeExtension">輸入前必須設定為false</param>
void RegionFloodFill(Mat& ImgBinary, int x, int y, vector<Point>& vectorPoint, vector<Point>& vContour,int maxArea,bool& isOverSizeExtension)
{
	if (maxArea < vectorPoint.size())
		return;

	uchar tagOverSize = 10;
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
				RegionFloodFill(ImgBinary, i, j, vectorPoint, vContour, maxArea, isOverSizeExtension);
			else if (ImgBinary.at<uchar>(j, i) == tagOverSize)
				isOverSizeExtension = true;
		}

#pragma endregion

	if (edgesSides > 1 && edgesSides < 8)
		vContour.push_back(Point(x, y));
}

void RegionPaint(Mat& ImgBinary, vector<Point> vPoint, uchar PaintIdx)
{
	for (int i = 0; i < vPoint.size(); i++)
		ImgBinary.at<uchar>(vPoint[i].y, vPoint[i].x) = PaintIdx;
}

vector<BlobInfo> RegionPartition(Mat ImgBinary,int maxArea, int minArea)
{
	vector<BlobInfo> lst;
	uchar tagOverSize = 10;

	Mat ImgTag = ImgBinary.clone();

	for (int i = 0; i < ImgBinary.cols; i++)
		for (int j = 0; j < ImgBinary.rows; j++)
		{
			int val = ImgTag.at<uchar>(j, i);
			bool isOverSizeExtension = false;

			if (val == 255)
			{
				vector<Point> vArea;
				vector<Point> vContour;
				RegionFloodFill(ImgTag, i, j, vArea, vContour, maxArea, isOverSizeExtension);

				if (vArea.size() > maxArea|| isOverSizeExtension)
				{
					RegionPaint(ImgTag, vArea, tagOverSize);
					continue;
				}
				else if (vArea.size() <= minArea)
				{
					RegionPaint(ImgTag, vArea, 0);
					continue;
				}

				if (vContour.size() == 0)
					continue;

				BlobInfo regionInfo = BlobInfo(vArea, vContour);

				RegionPaint(ImgTag, vArea, 0);

				lst.push_back(regionInfo);

			}

		}

	ImgTag.release();
	ImgBinary.release();
	return lst;

}



/// <summary>
/// 
/// </summary>
/// <param name="ImgBinary">必須輸入二值化影像</param>
/// <param name="filter">預先過濾條件之後想到會陸續增加</param>
/// <returns></returns>
vector<BlobInfo> RegionPartition(Mat ImgBinary, BlobFilter filter)
{

	float maxArea = INT_MAX-2;
	float minArea = -1;

	bool isEnable = filter.IsEnableArea();
	if (filter.IsEnableArea())
	{
		maxArea = filter.MaxArea();
		minArea = filter.MinArea();
	}

	float Xmin = 0;
	float Xmax = ImgBinary.cols;

	if (filter.IsEnableXbound())
	{
		Xmax = filter.MaxXbound();
		Xmin = filter.MinXbound();
	}

	float Ymin = 0;
	float Ymax = ImgBinary.rows;

	if (filter.IsEnableYbound())
	{
		Ymax = filter.MaxYbound();
		Ymin = filter.MinYbound();
	}

	vector<BlobInfo> lst;
	uchar tagOverSize = 10;

	Mat ImgTag = ImgBinary.clone();

	for (int i = Xmin; i < Xmax; i++)
		for (int j = Ymin; j < Ymax; j++)
		{
			int val = ImgTag.at<uchar>(j, i);
			bool isOverSizeExtension = false;

			if (val == 255)
			{
				vector<Point> vArea;
				vector<Point> vContour;
				RegionFloodFill(ImgTag, i, j, vArea, vContour, maxArea, isOverSizeExtension);

				if (vArea.size() > maxArea || isOverSizeExtension)
				{
					RegionPaint(ImgTag, vArea, tagOverSize);
					continue;
				}
				else if (vArea.size() <= minArea)
				{
					RegionPaint(ImgTag, vArea, 0);
					continue;
				}

				BlobInfo regionInfo = BlobInfo(vArea, vContour);

				RegionPaint(ImgTag, vArea, 0);

				lst.push_back(regionInfo);

			}

		}

	ImgTag.release();
	ImgBinary.release();
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
		float len =norm(_center - (Point2f)vContour[j]);

		if (len > max_len)
			max_len = len;

		if (len < min_len)
			min_len = len;
	}

	_circularity = _area / (max_len* max_len*CV_PI);

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

	_bulkiness = CV_PI * _Ra/2 * _Rb/2 / _area*1.0;


	if (minArea < _area)
		_rectangularity = minArea / _area;
	else
		_rectangularity = _area / minArea;

	_rectangularity = abs(_rectangularity);

	_compactness = (1.0 * _contour.size()) * (1.0 * _contour.size()) / (4.0 * CV_PI * _area);

	// _compactness 公式
	//
	// 
	//  _compactness= (周長)^2/ (4*PI*面積)
	//
	//  可以推算出 理論上 目標物的 _compactness 數值 在用此數值進行過濾
	//
	//

	// _roundness;
	// 

	if (_contour.size() > 0)
	{
		float distance=0;

		for (int i = 0; i < _contour.size(); i++)
		{
			float d = norm(_center - (Point2f)_contour[i]);
			distance += d;
		}
		distance /= _contour.size();

		float sigma;

		float diff = 0;

		for (int i = 0; i < _contour.size(); i++)
		{
			float d = norm(_center - (Point2f)_contour[i]);
			diff+=(d - distance)* (d - distance);
		}

		diff = sqrt(diff);

		sigma = diff / sqrt(_contour.size() * 1.0);
		_roundness = 1 - sigma / distance;
		_sides = (float)1.411 * pow((distance / sigma), (0.4724));
	}


	// Moments openCV已經存在實作 沒有必要加入此類特徵 有需要在呼叫即可
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

int BlobInfo::Xmin()
{
	return _XminBound;
}

int BlobInfo::Ymin()
{
	return _YminBound;
}

int BlobInfo::Xmax()
{
	return _XmaxBound;
}

int BlobInfo::Ymax()
{
	return _YmaxBound;
}

float BlobInfo::Bulkiness()
{
	return _bulkiness;
}

float BlobInfo::Compactness()
{
	return _compactness;
}

float BlobInfo::Roundness()
{
	return _roundness;
}

float BlobInfo::Sides()
{
	return _sides;
}

BlobFilter::BlobFilter()
{
	FilterCondition condition1;
	condition1.FeatureName = "area";
	condition1.Enable = false;
	condition1.MaximumValue = INT16_MAX;
	condition1.MinimumValue = INT16_MIN;

	FilterCondition condition2;
	condition2.FeatureName = "xBound";
	condition2.Enable = false;
	condition2.MaximumValue = INT16_MAX;
	condition2.MinimumValue = INT16_MIN;

	FilterCondition condition3;
	condition3.FeatureName = "yBound";
	condition3.Enable = false;
	condition3.MaximumValue = INT16_MAX;
	condition3.MinimumValue = INT16_MIN;

	FilterCondition condition4;
	condition4.FeatureName = "grayLevel";
	condition4.Enable = false;
	condition4.MaximumValue = 255;
	condition4.MinimumValue = 0;



	map.insert(std::pair<string, FilterCondition>(condition1.FeatureName, condition1));
	map.insert(std::pair<string, FilterCondition>(condition2.FeatureName, condition2));
	map.insert(std::pair<string, FilterCondition>(condition3.FeatureName, condition3));
}

BlobFilter::~BlobFilter()
{
	map.clear();
}

void BlobFilter::_setMaxPokaYoke(string title, float value)
{
	if (map[title].MinimumValue < value)
		map[title].MaximumValue = value;
	else
		map[title].MaximumValue = map[title].MinimumValue;
}

void BlobFilter::_setMinPokaYoke(string title, float value)
{
	if (map[title].MaximumValue > value)
		map[title].MinimumValue = value;
	else
		map[title].MinimumValue = map[title].MaximumValue;
}




bool BlobFilter::IsEnableArea()
{
	return map["area"].Enable;
}

float BlobFilter::MaxArea()
{
	return map["area"].MaximumValue;
}

float BlobFilter::MinArea()
{
	return map["area"].MinimumValue;
}

bool BlobFilter::IsEnableXbound()
{
	return map["xBound"].Enable;
}

float BlobFilter::MaxXbound()
{
	return map["xBound"].MaximumValue;
}

float BlobFilter::MinXbound()
{
	return map["xBound"].MinimumValue;
}

bool BlobFilter::IsEnableYbound()
{
	return map["yBound"].Enable;
}

float BlobFilter::MaxYbound()
{
	return map["yBound"].MaximumValue;
}

float BlobFilter::MinYbound()
{
	return map["yBound"].MinimumValue;
}

void BlobFilter::SetEnableArea(bool enable)
{
	map["area"].Enable = enable;
}

void BlobFilter::SetMaxArea(float value)
{
	_setMaxPokaYoke("area", value);
}

void BlobFilter::SetMinArea(float value)
{
	_setMinPokaYoke("area", value);
}

void BlobFilter::SetEnableXbound(bool enable)
{
	map["xBound"].Enable = enable;
}

void BlobFilter::SetMaxXbound(float value)
{
	_setMaxPokaYoke("xBound", value);
}

void BlobFilter::SetMinXbound(float value)
{
	_setMinPokaYoke("xBound", value);
}

void BlobFilter::SetEnableYbound(bool enable)
{
	map["yBound"].Enable = enable;
}

void BlobFilter::SetMaxYbound(float value)
{
	_setMaxPokaYoke("yBound", value);
}

void BlobFilter::SetMinYbound(float value)
{
	_setMinPokaYoke("yBound", value);
}

void BlobFilter::SetEnableGrayLevel(bool enable)
{
	map["grayLevel"].Enable = enable;
}

void BlobFilter::SetMaxGrayLevel(float value)
{
	if (value >= 0 && value <= 255 && value > map["grayLevel"].MinimumValue)
		map["grayLevel"].MaximumValue = (int)value;
	else
		map["grayLevel"].MaximumValue = map["grayLevel"].MinimumValue;
}

void BlobFilter::SetMinGrayLevel(float value)
{
	if (value >= 0 && value <= 255 && value < map["grayLevel"].MaximumValue)
		map["grayLevel"].MinimumValue = (int)value;
	else
		map["grayLevel"].MinimumValue = map["grayLevel"].MaximumValue;
}

