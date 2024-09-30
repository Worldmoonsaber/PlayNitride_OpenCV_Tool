
#include "OpenCV_Extension_Tool.h"

#pragma region BlobInfo 物件
BlobInfo::BlobInfo(vector<Point> vArea, vector<Point> vContour)
{
	CaculateBlob(vArea, vContour);
}

BlobInfo::BlobInfo()
{
}

BlobInfo::BlobInfo(Mat ImgRegion)
{
	vector<vector<Point>> vContourArr;
	findContours(ImgRegion, vContourArr, RETR_LIST, CHAIN_APPROX_NONE);
	vector<Point> vContour;
	
	for (int i = 0; i < vContourArr.size(); i++)
		vContour.insert(vContour.end(), vContourArr[i].begin(), vContourArr[i].end());

	vector<Point> vArea;
	findNonZero(ImgRegion, vArea);
	CaculateBlob(vArea, vContour);

	ImgRegion.release();
}

BlobInfo::BlobInfo(vector<Point> vContour)
{
	_contourMain = vContour;
	_contourHollow.clear();
	//retCOMP = cv::boundingRect(contH[i]);
	double areacomthres = cv::contourArea(vContour);

	_contour = vContour;
	//_points = vArea;
	_area = areacomthres;

	Moments M = (moments(vContour, false));
	_center = Point2f((M.m10 / M.m00), (M.m01 / M.m00));

	_XminBound = 99999;
	_YminBound = 99999;
	_XmaxBound = -1;
	_YmaxBound = -1;

	for (int i = 0; i < vContour.size(); i++)
	{
		if (vContour[i].x > _XmaxBound)
			_XmaxBound = vContour[i].x;

		if (vContour[i].y > _YmaxBound)
			_YmaxBound = vContour[i].y;

		if (vContour[i].x < _XminBound)
			_XminBound = vContour[i].x;

		if (vContour[i].y < _YminBound)
			_YminBound = vContour[i].y;

	}

	_Width = _XmaxBound - _XminBound + 1;
	_Height = _YmaxBound - _YminBound + 1;


	float min_len = _area;
	float max_len = 0;

	for (int j = 0; j < vContour.size(); j++)
	{
		float len = norm(_center - (Point2f)vContour[j]);

		if (len > max_len)
			max_len = len;

		if (len < min_len)
			min_len = len;
	}

	_circularity = _area / (max_len * max_len * CV_PI);

	if (vContour.size() == 0)
		return;

	RotatedRect minrect = minAreaRect(vContour);
	_Angle = minrect.angle;

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

	_bulkiness = CV_PI * _Ra / 2 * _Rb / 2 / _area * 1.0;

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
		float distance = 0;

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
			diff += (d - distance) * (d - distance);
		}

		diff = sqrt(diff);
		sigma = diff / sqrt(_contour.size() * 1.0);
		_roundness = 1 - sigma / distance;
		_sides = (float)1.411 * pow((distance / sigma), (0.4724));
	}

	// Moments openCV已經存在實作 沒有必要加入此類特徵 有需要在呼叫即可

}

BlobInfo::BlobInfo(vector<Point> vMainContour, vector<vector<Point>> vHollowContour)
{
	_contourMain = vMainContour;
	_contourHollow = vHollowContour;

	_area = cv::contourArea(vMainContour);
	Moments M = (moments(vMainContour, false));
	_center = Point2f((M.m10 / M.m00)*_area, (M.m01 / M.m00) * _area);

	for (int i = 0; i < vHollowContour.size(); i++)
	{
		M = (moments(vHollowContour[i], false));
		int sub_Area= contourArea(vHollowContour[i]);
		_center-= Point2f((M.m10 / M.m00) * sub_Area, (M.m01 / M.m00) * sub_Area);
		_area -= sub_Area;
	}

	_center /= _area;

	RotatedRect minrect = minAreaRect(vMainContour);
	_Angle = minrect.angle;

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

	_XminBound = 99999;
	_YminBound = 99999;
	_XmaxBound = -1;
	_YmaxBound = -1;

	for (int i = 0; i < vMainContour.size(); i++)
	{
		if (vMainContour[i].x > _XmaxBound)
			_XmaxBound = vMainContour[i].x;

		if (vMainContour[i].y > _YmaxBound)
			_YmaxBound = vMainContour[i].y;

		if (vMainContour[i].x < _XminBound)
			_XminBound = vMainContour[i].x;

		if (vMainContour[i].y < _YminBound)
			_YminBound = vMainContour[i].y;
	}

	_Width = _XmaxBound - _XminBound + 1;
	_Height = _YmaxBound - _YminBound + 1;

	_contour.clear();

	_contour.insert(_contour.end(), vMainContour.begin(), vMainContour.end());

	for (size_t i = 0; i < vHollowContour.size(); i++)
		_contour.insert(_contour.end(), vHollowContour[i].begin(), vHollowContour[i].end());

	vector<vector<Point>> vPoint;

	vPoint.push_back(vMainContour);

	for (int i = 0; i < vHollowContour.size(); i++)
		vPoint.push_back(vHollowContour[i]);

	float min_len = _area;
	float max_len = 0;

	for (int j = 0; j < _contour.size(); j++)
	{
		float len = norm(_center - (Point2f)_contour[j]);

		if (len > max_len)
			max_len = len;

		if (len < min_len)
			min_len = len;
	}

	_circularity = _area / (max_len * max_len * CV_PI);

	if (_contour.size() == 0)
		return;

	_bulkiness = CV_PI * _Ra / 2 * _Rb / 2 / _area * 1.0;

	if (minArea < _area)
		_rectangularity = minArea / _area;
	else
		_rectangularity = _area / minArea;

	_rectangularity = abs(_rectangularity);
	_compactness = (1.0 * _contour.size()) * (1.0 * _contour.size()) / (4.0 * CV_PI * _area);

	if (_contour.size() > 0)
	{
		float distance = 0;

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
			diff += (d - distance) * (d - distance);
		}

		diff = sqrt(diff);
		sigma = diff / sqrt(_contour.size() * 1.0);
		_roundness = 1 - sigma / distance;
		_sides = (float)1.411 * pow((distance / sigma), (0.4724));
	}


}

void BlobInfo::CaculateBlob(vector<Point> vArea, vector<Point> vContour)
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

	_Width = _XmaxBound - _XminBound + 1;
	_Height = _YmaxBound - _YminBound + 1;
	_center = Point2f(x_sum / vArea.size(), y_sum / vArea.size());

	float min_len = _area;
	float max_len = 0;

	for (int j = 0; j < vContour.size(); j++)
	{
		float len = norm(_center - (Point2f)vContour[j]);

		if (len > max_len)
			max_len = len;

		if (len < min_len)
			min_len = len;
	}

	_circularity = _area / (max_len * max_len * CV_PI);

	if (vContour.size() == 0)
		return;

	RotatedRect minrect = minAreaRect(vContour);
	_Angle = minrect.angle;

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

	_bulkiness = CV_PI * _Ra / 2 * _Rb / 2 / _area * 1.0;

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
		float distance = 0;

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
			diff += (d - distance) * (d - distance);
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
	_area = -1;
	_circularity = -1;
	_rectangularity = -1;
	_XminBound = -1;
	_YminBound = -1;
	_XmaxBound = -1;
	_YmaxBound = -1;
	_minRectWidth = -1;
	_minRectHeight = -1;
	_Angle = -1;
	_AspectRatio = -1;
	_Ra = -1;
	_Rb = -1;
	_bulkiness = -1;
	_compactness = -1;
	_roundness = -1;
	_sides = -1;
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

int BlobInfo::Width()
{
	return _Width;
}

int BlobInfo::Height()
{
	return _Height;
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

vector<vector<Point>> BlobInfo::contourHollow()
{
	return _contourHollow;
}

vector<Point> BlobInfo::contourMain()
{
	return _contourMain;
}

#pragma endregion

void RegionPartitionTopologySubLayerAnalysis(int layer,int curIndex, vector<vector<Point>> vContour, vector<Vec4i> vhi,vector<BlobInfo>& lstBlob)
{
	int type = layer % 2;
	//--- 0 此層為Region
	//--- 1 此層為挖空區

	if (type == 0)
	{
		//----沒有子階層
		vector<Point> mainContour = vContour[curIndex];
		vector<vector<Point>> vHollowContour;

		//---刪除子階層
		int idx = vhi[curIndex].val[2];
		vector<int> subIndx;

		if (idx != -1)
		{
			while (true)
			{
				vHollowContour.push_back(vContour[idx]);

				if (vhi[idx].val[2] != -1)
					subIndx.push_back(vhi[idx].val[2]);//--有Region
				if (vhi[idx].val[0] == -1)
					break;

				idx = vhi[idx].val[0];
			}
		}

		BlobInfo blob = BlobInfo(mainContour, vHollowContour);
		lstBlob.push_back(blob);

		for (int i = 0; i < subIndx.size(); i++)
			RegionPartitionTopologySubLayerAnalysis(layer + 1, subIndx[i], vContour, vhi, lstBlob);
	}
	else
	{
		//---挖空區域 觀察是否存在 子區域

		int idx = vhi[curIndex].val[2];
		vector<int> subIndx;

		if (idx != -1)
		{
			while (true)
			{
				if (vhi[idx].val[2] != -1)
					subIndx.push_back(vhi[idx].val[2]);

				if (vhi[idx].val[0] == -1)
					break;

				idx = vhi[idx].val[0];
			}
		}

		for (int i = 0; i < subIndx.size(); i++)
			RegionPartitionTopologySubLayerAnalysis(layer + 1, subIndx[i], vContour, vhi, lstBlob);

	}
}

vector<BlobInfo> RegionPartitionTopology(Mat ImgBinary)
{
	vector<BlobInfo> vRes;
	//https://blog.csdn.net/qinglingLS/article/details/106270095
	// 準備用拓樸的方式重構方法

	vector<vector<Point>> vContour;
	vector<Vec4i> vhi;
	//
	//  [下一個,上一個,子層,父層]
	//
	findContours(ImgBinary, vContour, vhi, RETR_CCOMP, CHAIN_APPROX_NONE);

	int layer = 0;
	int i = 0;

	if (vhi.size() > 0)
	{
		while (true)
		{
			if (vhi[i].val[2] == -1)
			{
				//----沒有子階層
				BlobInfo obj = BlobInfo(vContour[i]);
				vRes.push_back(obj);
			}
			else
			{
				//----有階層 待扣除坑洞區域
				RegionPartitionTopologySubLayerAnalysis(0, i, vContour, vhi, vRes);
			}

			if (vhi[i].val[0] == -1)
				break;

			i = vhi[i].val[0];
		}
	}

	return vRes;
}


void _GetRegionStatistics(vector<BlobInfo> vRegions,string Property, float& avg, float& std)
{
	if (vRegions.size() == 1)
	{
		if(Property=="Area")
			avg = vRegions[0].Area();
		else if (Property == "minRectWidth")
			avg = vRegions[0].minRectWidth();
		else if (Property == "minRectHeight")
			avg = vRegions[0].minRectHeight();
		else
			avg = vRegions[0].Area();

		std = 0;
		return;
	}
	else if (vRegions.size() == 0)
	{
		avg = 0;
		std = 0;
		return;
	}

	float sum = 0;

	for (int i = 0; i < vRegions.size(); i++)
	{
		if (Property == "Area")
			sum+= vRegions[i].Area();
		else if (Property == "minRectWidth")
			sum += vRegions[i].minRectWidth();
		else if (Property == "minRectHeight")
			sum += vRegions[i].minRectHeight();
		else
			sum += vRegions[i].Area();
	}

	avg = sum / vRegions.size();

	float sigma = 0;

	for (int i = 0; i < vRegions.size(); i++)
	{
		float element;// = vRegions[i].Area() - avg;

		if (Property == "Area")
			element = vRegions[i].Area() - avg;
		else if (Property == "minRectWidth")
			element = vRegions[i].minRectWidth() - avg;
		else if (Property == "minRectHeight")
			element = vRegions[i].minRectHeight() - avg;
		else
			element = vRegions[i].Area() - avg;

		sigma += element* element;
	}
	sigma /= vRegions.size();
	std = sqrt(sigma);
}


vector<BlobInfo> FilteredRegionByStatistics(vector<BlobInfo> vRegions)
{
	float avg_A; float std_A;
	_GetRegionStatistics(vRegions, "Area", avg_A, std_A);
	float avg_mW; float std_mW;
	_GetRegionStatistics(vRegions, "minRectWidth", avg_mW, std_mW);
	float avg_mH; float std_mH;
	_GetRegionStatistics(vRegions, "minRectHeight", avg_mH, std_mH);

	vector<BlobInfo> vRegionsNew;

	for (int i = 0; i < vRegions.size(); i++)
	{
		float value = 0;

		if (vRegions[i].Area() > avg_A + std_A || vRegions[i].Area() < avg_A - std_A)
			continue;

		if (vRegions[i].minRectWidth() > avg_mW + std_mW || vRegions[i].minRectWidth() < avg_mW - std_mW)
			continue;

		if (vRegions[i].minRectHeight() > avg_mH + std_mH || vRegions[i].minRectHeight() < avg_mH - std_mH)
			continue;
	
		vRegionsNew.push_back(vRegions[i]);
	
	}
	return vRegionsNew;
}


vector<tuple<Point, float>> MatchPattern(Mat Img, Mat MatchPattern, int div_x = 1, int div_y = 1, float Tolerance_score = 0.5)
{
	int xPatternGrid = div_x;
	int yPatternGrid = div_y;
	float tolerance_Score = Tolerance_score;

	if (xPatternGrid == 0)
		xPatternGrid = 1;

	if (yPatternGrid == 0)
		yPatternGrid = 1;

	vector<Point> vCropIndx;
	vector<Mat> vCropMatch;
	vector<Point> vCropCenter;

	int subWidth = MatchPattern.size().width / xPatternGrid;
	int subHeight = MatchPattern.size().height / yPatternGrid;

	for (int x = 0; x < xPatternGrid; x++)
		for (int y = 0; y < yPatternGrid; y++)
		{
			Mat croppedIMG;
			Rect rectPart = Rect(x * subWidth, y * subHeight, subWidth, subHeight);

			vCropIndx.push_back(Point(x, y));
			vCropCenter.push_back(Point((x + 0.5) * subWidth, (y + 0.5) * subHeight));
			MatchPattern(rectPart).copyTo(croppedIMG);
			vCropMatch.push_back(croppedIMG);
		}

	const int resSz = xPatternGrid * yPatternGrid;

	vector < vector<Point>> result;

	for (int i = 0; i < resSz; i++)
		result.push_back(vector<Point>());

	for (int i = 0; i < resSz; i++)
	{
		Mat imgMatched;
		matchTemplate(Img, vCropMatch[i], imgMatched, TM_CCOEFF_NORMED);

		Mat dst;
		threshold(imgMatched, dst, tolerance_Score, 1, THRESH_BINARY);

		Mat dstBinary = Mat::zeros(dst.size(), CV_8UC1);

		//-----------------預期會有多點符合條件
		for (int i = 0; i < dst.cols; ++i)
			for (int j = 0; j < dst.rows; ++j)
				if (dst.at<float>(j, i) != 0)
				{
					dstBinary.at<uchar>(j, i) = 255;
				}

		vector<BlobInfo> vRegionTmp = RegionPartitionTopology(dstBinary);
		vector<BlobInfo> vRegion;


		if (vRegionTmp.size() > 0)
		{
			float avg_Area = 0, std_Area = 0;

			//-----刪除 Noise 
			vRegion=FilteredRegionByStatistics(vRegionTmp);
		}
		else
		{
			vRegion.push_back(vRegionTmp[0]);
		}



		Point offset = Point(vCropMatch[i].size().width / 2 - 1, vCropMatch[i].size().height / 2 - 1);

		for (int j = 0; j < vRegion.size(); j++)
		{
			Point pt = Point(vRegion[j].Center()) + offset;
			result[i].insert(result[i].begin(), pt);
		}

		dst.release();
		imgMatched.release();
		dstBinary.release();
	}

	vector<tuple<Point, float>> vMatchResult;

	if (result.size() == 1)
	{
		//----只有一個Template 無法計算角度
		for (int i = 0; i < result[0].size(); i++)
		{
			vMatchResult.push_back(tuple<Point, float>(result[0][i], 0.0));
		}
	}
	else
	{
		vector<vector<Point>> matchedComfirm;
		//----一一比對
		for (int i = 0; i < result[0].size(); i++)
		{
			vector<Point> vSet;
			vSet.push_back(result[0][i]);//參考基準點一

			Point ref_Pt = result[0][i];

			for (int s = 1; s < result.size(); s++)
			{
				if (result[s].size() == 0)
					break;

				float x_diff_ref = vCropCenter[s].x - vCropCenter[0].x;
				float y_diff_ref = vCropCenter[s].y - vCropCenter[0].y;

				Point ptDiff = Point(x_diff_ref, y_diff_ref);

				float ptDiffDist = norm(ptDiff);

				std::sort(result[s].begin(), result[s].end(), [&, ref_Pt, x_diff_ref, y_diff_ref](Point& a, Point& b)
					{
						Point pointDist_A = a - ref_Pt;
						Point pointDist_B = b - ref_Pt;

						if (norm(pointDist_A) < norm(pointDist_B))
						{
							return true;
						}
						return false;
					});

				int idx = 0;

				//判斷是否合理
				Point pointDist = result[s][0] - ref_Pt;
				float dist = norm(pointDist);

				if (ptDiffDist * 1.1 < dist)
				{
					break;
				}

				//-----計算向量

				float arcCos = (ptDiff.x* pointDist.x+ ptDiff.y * pointDist.y)/ (1.0*(norm(ptDiff) * norm(pointDist)));
				float angle = acos(arcCos) * 180.0 / CV_PI;

				//std::cout << "arcCos:: " << arcCos << endl;
				//std::cout << "angle:: " << angle << endl;

				//------待加入其他糾錯條件
				//------目前條件過於單薄 穩定性不足

				vSet.push_back(result[s][0]);//參考基準點一
				result[s].erase(result[s].begin());
			}

			if (vSet.size()== result.size() && vSet.size() > 1)
			{
				matchedComfirm.push_back(vSet);
			}
		}

		for (int i = 0; i < matchedComfirm.size(); i++)
		{
			if (matchedComfirm[i].size() == 0)
				continue;

			RotatedRect minrect = minAreaRect(matchedComfirm[i]);
			vMatchResult.push_back(tuple<Point, float>(minrect.center, minrect.angle));
		}
	}

	vCropIndx.clear();
	vCropCenter.clear();

	for (int i = 0; i < vCropMatch.size(); i++)
		vCropMatch[i].release();

	vCropMatch.clear();

	return vMatchResult;
}

vector<tuple<Point, float>> MatchPattern(Mat Img, Mat MatchPatternImg, float Tolerance_score)
{
	int div_x = 1;
	int div_y = 1;

	float ratio=1;

	if (MatchPatternImg.cols > MatchPatternImg.rows)
		ratio = MatchPatternImg.cols*1.0 / MatchPatternImg.rows*1.0;
	else
		ratio = MatchPatternImg.rows*1.0 / MatchPatternImg.cols*1.0;

	ratio = round(ratio);

	if (ratio >= 2)
	{
		if (MatchPatternImg.cols > MatchPatternImg.rows)
		{
			div_x = (int)ratio;
			div_y = 1;
		}
		else
		{
			div_x = 1;
			div_y = (int)ratio;
		}

	}

	return MatchPattern(Img, MatchPatternImg, div_x, div_y, Tolerance_score);
}


//--------------------Match Object

//MatchObject::MatchObject(Mat Pattern)
//{
//}
