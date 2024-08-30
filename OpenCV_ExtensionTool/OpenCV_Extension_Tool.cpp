
#include "OpenCV_Extension_Tool.h"

#pragma region Region 標記 功能 內部方法

void RegionFloodFill(uchar* ptr, int x, int y, vector<Point>& vectorPoint, vector<Point>& vContour, int maxArea, bool& isOverSizeExtension,int width,int height)
{
	if (maxArea < vectorPoint.size())
		return;

	uchar tagOverSize = 10;
	uchar tagIdx = 101;

	if (ptr[x + width * y] == 255)
	{
		ptr[x + width * y] = tagIdx;
		vectorPoint.push_back(Point(x, y));
	}
	else if (ptr[x + width * y] == tagIdx)
		return;

	uchar edgesSides = 0; //降低佔據空間

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

			if (i >= width || j >= height)
			{
				edgesSides++;
				continue;
			}

			if (ptr[i + width * j] == 0)
			{
				edgesSides++;
				continue;
			}

			if (ptr[i + width * j] == 255)
				RegionFloodFill(ptr, i, j, vectorPoint, vContour, maxArea, isOverSizeExtension, width, height);
			else if (ptr[i + width * j] == tagOverSize)
				isOverSizeExtension = true;
			else if (ptr[i + width * j] == tagIdx)
				continue;

		}

#pragma endregion

	if (edgesSides > 1 && edgesSides < 8)
		vContour.push_back(Point(x, y));
}

void RegionPaint(uchar* ptr, vector<Point> vPoint, uchar PaintIdx,int imgWidth)
{
	//<---此處加入平行運算需要額外的函式庫, UI端電腦環境不明,先不實作此類內容
	for (int i = 0; i < vPoint.size(); i++)
		ptr[vPoint[i].y * imgWidth + vPoint[i].x] = PaintIdx;
}

#pragma endregion

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

#pragma endregion

#pragma region BlobFilter 物件

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



	mapConditions.insert(std::pair<string, FilterCondition>(condition1.FeatureName, condition1));
	mapConditions.insert(std::pair<string, FilterCondition>(condition2.FeatureName, condition2));
	mapConditions.insert(std::pair<string, FilterCondition>(condition3.FeatureName, condition3));
	mapBool.insert(std::pair<string, bool>("SubRegion", true));
}

BlobFilter::~BlobFilter()
{
	mapConditions.clear();
}

void BlobFilter::_setMaxPokaYoke(string title, float value)
{
	if (mapConditions[title].MinimumValue < value)
		mapConditions[title].MaximumValue = value;
	else
		mapConditions[title].MaximumValue = mapConditions[title].MinimumValue;
}

void BlobFilter::_setMinPokaYoke(string title, float value)
{
	if (mapConditions[title].MaximumValue > value)
		mapConditions[title].MinimumValue = value;
	else
		mapConditions[title].MinimumValue = mapConditions[title].MaximumValue;
}

bool BlobFilter::IsEnableArea()
{
	return mapConditions["area"].Enable;
}

float BlobFilter::MaxArea()
{
	return mapConditions["area"].MaximumValue;
}

float BlobFilter::MinArea()
{
	return mapConditions["area"].MinimumValue;
}

bool BlobFilter::IsEnableXbound()
{
	return mapConditions["xBound"].Enable;
}

float BlobFilter::MaxXbound()
{
	return mapConditions["xBound"].MaximumValue;
}

float BlobFilter::MinXbound()
{
	return mapConditions["xBound"].MinimumValue;
}

bool BlobFilter::IsEnableYbound()
{
	return mapConditions["yBound"].Enable;
}

float BlobFilter::MaxYbound()
{
	return mapConditions["yBound"].MaximumValue;
}

float BlobFilter::MinYbound()
{
	return mapConditions["yBound"].MinimumValue;
}

bool BlobFilter::IsEnableSubRegion()
{
	return mapBool["SubRegion"];
}

void BlobFilter::SetEnableArea(bool enable)
{
	mapConditions["area"].Enable = enable;
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
	mapConditions["xBound"].Enable = enable;
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
	mapConditions["yBound"].Enable = enable;
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
	mapConditions["grayLevel"].Enable = enable;
}

void BlobFilter::SetMaxGrayLevel(float value)
{
	if (value >= 0 && value <= 255 && value > mapConditions["grayLevel"].MinimumValue)
		mapConditions["grayLevel"].MaximumValue = (int)value;
	else
		mapConditions["grayLevel"].MaximumValue = mapConditions["grayLevel"].MinimumValue;
}

void BlobFilter::SetMinGrayLevel(float value)
{
	if (value >= 0 && value <= 255 && value < mapConditions["grayLevel"].MaximumValue)
		mapConditions["grayLevel"].MinimumValue = (int)value;
	else
		mapConditions["grayLevel"].MinimumValue = mapConditions["grayLevel"].MaximumValue;
}

void BlobFilter::SetEnableSubRegion(bool enable)
{
	mapBool["SubRegion"] = enable;
}

#pragma endregion

#pragma region BlobInfoThreadObject 物件

/// <summary>
/// 用於多緒處理 BlobInfo物件 提升效率用
/// </summary>
class BlobInfoThreadObject
{
public:
	BlobInfoThreadObject();
	void Initialize();
	void AddObject(vector<Point> vArea, vector<Point> vContour);
	void WaitWorkDone();
	vector<BlobInfo> GetObj();
	~BlobInfoThreadObject();
private:
	static void thread_WorkContent(queue <tuple<vector<Point>, vector<Point>>>* queue, bool* _isFinished, vector<BlobInfo>* vResult, mutex* mutex);
	thread thread_Work;
	queue <tuple<vector<Point>, vector<Point>>> _QueueObj;
	vector<BlobInfo> _result;
	bool _isProcessWaitToFinished = false;
	mutex mu;
};

BlobInfoThreadObject::BlobInfoThreadObject()
{
	_isProcessWaitToFinished = false;
}

void BlobInfoThreadObject::Initialize()
{
	thread_Work = thread(BlobInfoThreadObject::thread_WorkContent,&_QueueObj,&_isProcessWaitToFinished,&_result,&mu);
}

void BlobInfoThreadObject::AddObject(vector<Point> vArea, vector<Point> vContour)
{
	mu.lock();
	_QueueObj.push(tuple<vector<Point>, vector<Point>>(vArea, vContour));
	mu.unlock();
}

void BlobInfoThreadObject::WaitWorkDone()
{
	_isProcessWaitToFinished = true;
	thread_Work.join();
}

vector<BlobInfo> BlobInfoThreadObject::GetObj()
{
	return _result;
}

BlobInfoThreadObject::~BlobInfoThreadObject()
{
	thread_Work.~thread();
}

void BlobInfoThreadObject::thread_WorkContent(queue <tuple<vector<Point>, vector<Point>>>* queue, bool* _isFinished, vector<BlobInfo>* vResult,mutex* mutex)
{
	while (true)
	{
		if (queue->size() != 0)
		{
			tuple<vector<Point>, vector<Point>> obj;

			mutex->lock();
			obj = queue->front();//取得資料
			queue->pop();//將資料從_QueueObj中清除		
			mutex->unlock();

			vector<Point> vArea = std::get<0>(obj);
			vector<Point> vContour = std::get<1>(obj);
			BlobInfo regionInfo = BlobInfo(vArea, vContour);
			vResult->push_back(regionInfo);
		}

		if (_isFinished[0] == true && queue->size() == 0)
			break;//工作完成
	}
}

#pragma endregion

vector<BlobInfo> RegionPartition(Mat ImgBinary,int maxArea, int minArea)
{
	vector<BlobInfo> lst;
	lst.reserve(100);
	uchar tagOverSize = 10;
	Mat ImgTag = ImgBinary.clone();

	uchar* _ptr = (uchar*)ImgTag.data;
	int ww = ImgBinary.cols;
	int hh = ImgBinary.rows;

	BlobInfoThreadObject blobInfoThread;
	blobInfoThread.Initialize();

	for (int i = 0; i < ww; i++)
		for (int j = 0; j < hh; j++)
		{
			uchar val = _ptr[ww * j + i];
			bool isOverSizeExtension = false;

			if (val == 255)
			{
				vector<Point> vArea;
				vector<Point> vContour;
				RegionFloodFill(_ptr, i, j, vArea, vContour, maxArea, isOverSizeExtension,ww,hh);

				if (vArea.size() > maxArea || isOverSizeExtension)
				{
					RegionPaint(_ptr, vArea, tagOverSize, ww);
					continue;
				}
				else if (vArea.size() <= minArea)
				{
					RegionPaint(_ptr, vArea, 0, ww);
					continue;
				}
				blobInfoThread.AddObject(vArea, vContour);
				RegionPaint(_ptr, vArea, 0, ww);
			}
		}

	blobInfoThread.WaitWorkDone();
	lst = blobInfoThread.GetObj();

	ImgTag.release();
	ImgBinary.release();
	return lst;

}

vector<BlobInfo> RegionPartition(Mat ImgBinary)
{
	return RegionPartition(ImgBinary, INT16_MAX, 0);
}

/// <summary>
/// 
/// </summary>
/// <param name="ImgBinary">必須輸入二值化影像</param>
/// <param name="filter">預先過濾條件之後想到會依照需求陸續增加</param>
/// <returns></returns>
vector<BlobInfo> RegionPartition(Mat ImgBinary, BlobFilter filter)
{
	float maxArea = INT_MAX-2;
	float minArea = 0;

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

	uchar* _ptr = (uchar*)ImgTag.data;
	int ww = ImgBinary.cols;
	int hh = ImgBinary.rows;

	BlobInfoThreadObject blobInfoThread;
	blobInfoThread.Initialize();

	for (int i = Xmin; i < Xmax; i++)
		for (int j = Ymin; j < Ymax; j++)
		{
			uchar val = _ptr[ww * j + i];
			bool isOverSizeExtension = false;

			if (val == 255)
			{
				vector<Point> vArea;
				vector<Point> vContour;
				RegionFloodFill(_ptr, i, j, vArea, vContour, maxArea, isOverSizeExtension, ww, hh);

				if (vArea.size() > maxArea || isOverSizeExtension)
				{
					RegionPaint(_ptr, vArea, tagOverSize, ww);
					continue;
				}
				else if (vArea.size() <= minArea)
				{
					RegionPaint(_ptr, vArea, 0, ww);
					continue;
				}
				blobInfoThread.AddObject(vArea, vContour);
				RegionPaint(_ptr, vArea, 0, ww);
			}
		}

	blobInfoThread.WaitWorkDone();
	lst = blobInfoThread.GetObj();

	ImgTag.release();
	ImgBinary.release();
	return lst;
}

vector<BlobInfo> RegionPartitionNonMultiThread(Mat ImgBinary, int maxArea, int minArea)
{
	vector<BlobInfo> lst;
	lst.reserve(100);
	uchar tagOverSize = 10;
	Mat ImgTag = ImgBinary.clone();

	uchar* _ptr = (uchar*)ImgTag.data;
	int ww = ImgBinary.cols;
	int hh = ImgBinary.rows;


	for (int i = 0; i < ww; i++)
		for (int j = 0; j < hh; j++)
		{
			uchar val = _ptr[ww * j + i];
			bool isOverSizeExtension = false;

			if (val == 255)
			{
				vector<Point> vArea;
				vector<Point> vContour;
				RegionFloodFill(_ptr, i, j, vArea, vContour, maxArea, isOverSizeExtension, ww, hh);

				if (vArea.size() > maxArea || isOverSizeExtension)
				{
					RegionPaint(_ptr, vArea, tagOverSize, ww);
					continue;
				}
				else if (vArea.size() <= minArea)
				{
					RegionPaint(_ptr, vArea, 0, ww);
					continue;
				}
				BlobInfo info= BlobInfo(vArea, vContour);
				lst.push_back(info);
				RegionPaint(_ptr, vArea, 0, ww);
			}
		}

	ImgTag.release();
	ImgBinary.release();

	return lst;
}

vector<BlobInfo> RegionPartitionNonMultiThread(Mat ImgBinary)
{
	return RegionPartitionNonMultiThread(ImgBinary,INT16_MAX,0);
}


void RegionPartitionTopologySubLayerAnalysis(cv::Size sz,int layer,int curIndex, vector<vector<Point>> vContour, vector<Vec4i> vhi,vector<BlobInfo>& lstBlob)
{
	int type = layer % 2;
	//--- 0 此層為Region
	//--- 1 此層為挖空區

	if (type == 0)
	{
		Mat img = Mat(sz, CV_8UC1);
		//----沒有子階層
		fillConvexPoly(img, vContour[curIndex], Scalar(255, 255, 255));

		//---刪除子階層
		int idx = vhi[curIndex].val[2];
		vector<int> subIndx;

		if (idx != -1)
		{
			while (true)
			{
				fillConvexPoly(img, vContour[idx], Scalar(0, 0, 0));

				if (vhi[idx].val[2] != -1)
					subIndx.push_back(vhi[idx].val[2]);

				if (vhi[idx].val[0] == -1)
					break;

				idx = vhi[idx].val[0];
			}
		}

		BlobInfo blob = BlobInfo(img);
		lstBlob.push_back(blob);

		for (int i = 0; i < subIndx.size(); i++)
			RegionPartitionTopologySubLayerAnalysis(sz, layer + 1, subIndx[i], vContour, vhi, lstBlob);
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
			RegionPartitionTopologySubLayerAnalysis(sz, layer + 1, subIndx[i], vContour, vhi, lstBlob);

	}
}


/// <summary>
///  實測結果比較慢 (理論上應該要比較快) 待釐清
/// </summary>
/// <param name="ImgBinary"></param>
/// <param name="filter"></param>
/// <returns></returns>
vector<BlobInfo> RegionPartitionTopology(Mat ImgBinary, BlobFilter filter)
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
	while (true)
	{
		if (vhi[i].val[2] == -1)
		{
			Mat img = Mat(ImgBinary.size(), CV_8UC1);
			//----沒有子階層
			fillConvexPoly(img, vContour[i], Scalar(255, 255, 255));
			BlobInfo blob = BlobInfo(img);
			vRes.push_back(blob);
			img.release();
		}
		else
		{
			//----有階層 待扣除坑洞區域
			RegionPartitionTopologySubLayerAnalysis(ImgBinary.size(),0, i, vContour, vhi, vRes);
		}


		if (vhi[i].val[0] == -1)
			break;

		i = vhi[i].val[0];
	}

	return vRes;
}






void thread_Content(Mat* img, int maxArea, int minArea, int starY, int endY , vector<BlobInfo>* vResult, vector<BlobInfo>* vEdge)
{
	//vector<BlobInfo> lst;
	uchar tagOverSize = 10;
	Mat ImgTag = img->clone();

	uchar* _ptr = (uchar*)ImgTag.data;
	int ww = img->cols;
	int hh = img->rows;

	BlobInfoThreadObject blobInfoThread;
	blobInfoThread.Initialize();

	for (int i = 0; i < ww; i++)
		for (int j = starY; j < endY; j++)
		{
			uchar val = _ptr[ww * j + i];
			bool isOverSizeExtension = false;

			if (val == 255)
			{
				vector<Point> vArea;
				vector<Point> vContour;
				RegionFloodFill(_ptr, i, j, vArea, vContour, maxArea, isOverSizeExtension, ww, hh);

				if (vArea.size() > maxArea || isOverSizeExtension)
				{
					RegionPaint(_ptr, vArea, tagOverSize, ww);
					continue;
				}
				else if (vArea.size() <= minArea)
				{
					RegionPaint(_ptr, vArea, 0, ww);
					continue;
				}
				blobInfoThread.AddObject(vArea, vContour);
				RegionPaint(_ptr, vArea, 0, ww);
			}
		}

	blobInfoThread.WaitWorkDone();
	vResult[0] = blobInfoThread.GetObj();

}


//--經過初步測試效益不高
//vector<BlobInfo> RegionPartition2(Mat ImgBinary)
//{
//	vector<BlobInfo> result;
//
//	vector<thread> vThread;
//
//	Mat img = ImgBinary.clone();
//	
//	int max = INT16_MAX;
//	int min = INT16_MIN;
//
//	int mid = (int)(0.5 * (img.rows));
//
//
//	vector<BlobInfo> vResult1; 
//	vector<BlobInfo> vEdge1;
//	vector<BlobInfo> vResult2; 
//	vector<BlobInfo> vEdge2;
//	//thread thr1(thread_Content, &ImgBinary, INT16_MAX, INT16_MIN, 0, mid, &vResult1, &vEdge1);
//
//	int endY = img.rows;
//	thread thr2(thread_Content, &ImgBinary, INT16_MAX, 0,mid,endY, &vResult2, &vEdge2);
//
//	//thr1.join();
//	thr2.join();
//
//	return vResult2;
//}
