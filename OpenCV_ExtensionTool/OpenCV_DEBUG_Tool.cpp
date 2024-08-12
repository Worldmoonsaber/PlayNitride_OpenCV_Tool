#include "OpenCV_DEBUG_Tool.h"

void readFileParameter(string path, bool& isReadFileSucceed, vector<float>& fParamArr)
{
    ifstream in;

    in.open(path);
    if (in.fail()) 
    {
        isReadFileSucceed = false;
        in.close();
        return;
    }

    int index = 0;
    float value = 0;

    vector<string> vStr;
    string line;
    while (getline(in, line))
        vStr.push_back(line);

    for (int i = 0; i < vStr.size(); i++)
    {
        float val_f = atof(vStr[i].c_str());
        fParamArr.push_back(val_f);
    }

    in.close();
}

void saveFileParameter(string path, bool& isReadFileSucceed, vector<float> fParamArr)
{
    isReadFileSucceed = true;
    ofstream out;
    out.open(path);

    if (out.is_open())
    {
        for (int i = 0; i < fParamArr.size(); i++)
        {
            out << fParamArr[i] << endl;
        }
        out.close();
    }

}

void GetAllFolderBmpImage(string FolderPath, vector<string>& vStr)
{
    intptr_t   hFile = 0;
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(FolderPath).append("\\*.bmp").c_str(), &fileinfo)) != -1)//assign方法可以理解為先將原字串清空，然後賦予新的值作替換。
    {
        do
        {
            if ((fileinfo.attrib & _A_SUBDIR))//是否為資料夾
            {
                //----Do Nothing
            }
            else//非資料夾
            {
                vStr.push_back(p.assign(FolderPath).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }

}


#pragma region debugWindow 類別

class debugWindow
{
public:
    debugWindow();
    ~debugWindow();
    void Run(Mat imageInput, vector<BlobInfo> BlobInfo);
    void onMouse(int event, int x, int y, int flag);
private:
    static void onMouseStatic(int event, int x, int y, int flag, void* obj);
    void updateDisplayImage();
    Mat image1;
    Mat image2;
    Mat imageProcessing;
    Mat displayImage;
    cv::Point2f offset;
    Rect roiRect;
    float zoomFactor;
    string strTitle;
    vector<BlobInfo> vBlobList;
    int selectBlobIndex;
    int selectImageIndex;
    void doKeyAction(int key);
    bool isNeedRefresh;
};

debugWindow::debugWindow()
{
    selectBlobIndex = -1;
    offset = Point2f(0, 0);
    zoomFactor = 1.0;
    strTitle = "顯示圖片";
    isNeedRefresh = false;
}

debugWindow::~debugWindow()
{
    selectBlobIndex = -1;
    vBlobList.clear();
    image1.release();
    displayImage.release();
}

void debugWindow::onMouseStatic(int event, int x, int y, int flag, void* obj)
{
    debugWindow* dbw = static_cast<debugWindow*>(obj);
    dbw->onMouse(event, x, y, flag);
}

void debugWindow::onMouse(int event, int x, int y, int flag)
{
    int imgX = static_cast<int>(roiRect.x + (x) / zoomFactor);
    int imgY = static_cast<int>(roiRect.y + (y) / zoomFactor);


    Point ptLU = Point(roiRect.x, roiRect.y);
    Point ptRD = Point(roiRect.x+ roiRect.width, roiRect.y+ roiRect.height);



    if (event == cv::EVENT_RBUTTONDOWN) 
    {
        if (imgX >= 0 && imgX < image1.cols && imgY >= 0 && imgY < image1.rows)
        {
            bool isFound = false;
            int _selectIndex = -1;
            //---搜索範圍
            for (int i = 0; i < vBlobList.size(); i++)
            {
                if (vBlobList[i].Xmax() < imgX || vBlobList[i].Xmin() > imgX)
                    continue;

                if (vBlobList[i].Ymax() < imgY || vBlobList[i].Ymin() > imgY)
                    continue;

                Point ptRef = Point(imgX, imgY);

               for(int j=0;j< vBlobList[i].Points().size();j++)
                   if (vBlobList[i].Points()[j] == ptRef)
                   {
                       _selectIndex = i;
                       break;
                   }

               if (_selectIndex != -1)
                   break;
            }

            selectBlobIndex = _selectIndex;
            isNeedRefresh = true;
        }
    }
    else if (event == cv::EVENT_MOUSEHWHEEL)
    {
        //int delta = getMouseWheelDelta(flag);

        //if(delta>0)
        //    zoomFactor *= 1.05;
        //else
        //    zoomFactor /= 1.05;

        //int d_left=imgX- roiRect.x;
        //int d_right= roiRect.x+ roiRect.width- imgX;
        //int d_up= imgY- roiRect.y;
        //int d_down= roiRect.y + roiRect.height - imgY;

        //int ww=(d_left+ d_right) / zoomFactor;
        //int hh = (d_up + d_down) / zoomFactor;

        //roiRect = Rect(imgX - d_left / zoomFactor, imgY - d_up / zoomFactor,ww ,hh );
        //resize(image1, image1, Size(), zoomFactor, zoomFactor);

    }
    else if (event == cv::EVENT_MBUTTONDBLCLK)
    {
        //zoomFactor /= 1.1;

        //int d_left = imgX - roiRect.x;
        //int d_right = roiRect.x + roiRect.width - imgX;
        //int d_up = imgY - roiRect.y;
        //int d_down = roiRect.y + roiRect.height - imgY;

        //int ww = (d_left + d_right) / zoomFactor;
        //int hh = (d_up + d_down) / zoomFactor;

        //roiRect = Rect(imgX - d_left / zoomFactor, imgY - d_up / zoomFactor, ww, hh);

        //zoomFactor /= 1.1;


        //if (zoomFactor < 1)
        //{
        //    zoomFactor = 1;
        //    roiRect = Rect(0, 0, image1.cols, image1.rows);
        //}
    }

}

void debugWindow::updateDisplayImage()
{

    if (!imageProcessing.empty())
        imageProcessing.release();

    imageProcessing = image1.clone();

    if (selectBlobIndex != -1)
    {
        // P.S.: 不能用drawContours 因為Harvie演算法抓出來的Contour不是連續,是隨機排列, 用drawContours 畫出來會很奇怪 
        int channels=imageProcessing.channels();

        for (int i = 0; i < vBlobList[selectBlobIndex].contour().size(); i++)
        {
            if (channels == 3)
                imageProcessing.at<Vec3b>(vBlobList[selectBlobIndex].contour()[i].y, vBlobList[selectBlobIndex].contour()[i].x) = Vec3b(0, 0, 255);
            else if (channels == 4)
                imageProcessing.at<Vec4b>(vBlobList[selectBlobIndex].contour()[i].y, vBlobList[selectBlobIndex].contour()[i].x) = Vec4b(0, 0, 255, 255);
            else
                imageProcessing.at<uchar>(vBlobList[selectBlobIndex].contour()[i].y, vBlobList[selectBlobIndex].contour()[i].x) = 100;
        }

        if (isNeedRefresh)
        {
            system("cls");

            std::cout << "--------選擇圖塊特徵----------" << endl;
            std::cout << "面積          :" + to_string(vBlobList[selectBlobIndex].Area()) << endl;
            std::cout << "長寬比        :" + to_string(vBlobList[selectBlobIndex].AspectRatio()) << endl;
            std::cout << "Circularity   :" + to_string(vBlobList[selectBlobIndex].Circularity()) << endl;
            std::cout << "Rectangularity:" + to_string(vBlobList[selectBlobIndex].Rectangularity()) << endl;
            std::cout << "Bulkiness     :" + to_string(vBlobList[selectBlobIndex].Bulkiness()) << endl;
            std::cout << "Compactness   :" + to_string(vBlobList[selectBlobIndex].Compactness()) << endl;
            std::cout << "長軸          :" + to_string(vBlobList[selectBlobIndex].Ra()) << endl;
            std::cout << "短軸          :" + to_string(vBlobList[selectBlobIndex].Rb()) << endl;
            std::cout << "Roundness     :" + to_string(vBlobList[selectBlobIndex].Roundness()) << endl;
            std::cout << "Sides         :" + to_string(vBlobList[selectBlobIndex].Sides()) << endl;
        }

    }
    else
    {
        if (isNeedRefresh)
        {
            system("cls");
            std::cout << "請在視窗上點選滑鼠右鍵選擇 已被標籤的圖塊" << endl;
        }
    }


    Mat m_ROI = imageProcessing(roiRect);

    float sz = image1.cols * 1.0 / m_ROI.cols*1.0;

    cv::resize(m_ROI, displayImage, cv::Size(), sz, sz);

    cv::imshow(strTitle, displayImage);
    isNeedRefresh = false;
}

void debugWindow::doKeyAction(int key)
{
    if (key == '1')
        selectImageIndex = 1;
    else if (key == '2')
        selectImageIndex = 2;

    if (key == '+')
        zoomFactor = 1.05;
    else if (key == '-')
        zoomFactor /=1.05;

    if (key=='+' || key=='-')
    {
        //Point center = Point(roiRect.x + roiRect.width / 2, roiRect.y + roiRect.height / 2);

        //float sz;
        //if (key == '+')
        //    sz = 1.05;
        //else
        //    sz = 1/1.05;


        //int ww = roiRect.width * sz;
        //int hh = roiRect.height * sz;


        //roiRect = Rect(center.x - ww / 2, center.y - hh / 2, ww, hh);
        //int d_left=imgX- roiRect.x;
//int d_right= roiRect.x+ roiRect.width- imgX;
//int d_up= imgY- roiRect.y;
//int d_down= roiRect.y + roiRect.height - imgY;

//int ww=(d_left+ d_right) / zoomFactor;
//int hh = (d_up + d_down) / zoomFactor;

//roiRect = Rect(imgX - d_left / zoomFactor, imgY - d_up / zoomFactor,ww ,hh );
//resize(image1, image1, Size(), zoomFactor, zoomFactor);

    }


}

void debugWindow::Run(Mat imageInput, vector<BlobInfo> BlobInfo)
{
    system("cls");

    vBlobList = BlobInfo;

    image1 = imageInput.clone();
    cv::namedWindow(strTitle);
    cv::setMouseCallback("顯示圖片", debugWindow::onMouseStatic, &image1);
    roiRect = Rect(0,0, image1.cols,image1.rows);

    while (true) {
        updateDisplayImage();
        int key = cv::waitKey(10);
        if (key == 27) break; // 按下ESC鍵退出

        doKeyAction(key);
        //if (key == '+') scale *= 1.1; // 放大
        //if (key == '-') scale /= 1.1; // 縮小
        //if (key == 'w') offset.y -= 500; // 上移
        //if (key == 's') offset.y += 500; // 下移
        //if (key == 'a') offset.x -= 500; // 左移
        //if (key == 'd') offset.x += 500; // 右移
    }
}

#pragma endregion


void ShowDebugWindow(Mat Img_Ref, vector<BlobInfo> BlobInfo)
{
    debugWindow dbW = debugWindow();
    dbW.Run(Img_Ref, BlobInfo);
    dbW.~debugWindow();
}


