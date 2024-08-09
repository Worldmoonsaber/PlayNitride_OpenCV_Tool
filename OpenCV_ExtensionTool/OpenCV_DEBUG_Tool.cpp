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
    if ((hFile = _findfirst(p.assign(FolderPath).append("\\*.bmp").c_str(), &fileinfo)) != -1)//assign��k�i�H�z�Ѭ����N��r��M�šA�M��ᤩ�s���ȧ@�����C
    {
        do
        {
            if ((fileinfo.attrib & _A_SUBDIR))//�O�_����Ƨ�
            {
                //----Do Nothing
            }
            else//�D��Ƨ�
            {
                vStr.push_back(p.assign(FolderPath).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }

}


#pragma region debugWindow ���O

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
    strTitle = "��ܹϤ�";
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

    if (event == cv::EVENT_RBUTTONDOWN) 
    {
        if (imgX >= 0 && imgX < image1.cols && imgY >= 0 && imgY < image1.rows)
        {
            bool isFound = false;
            int _selectIndex = -1;
            //---�j���d��
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
    else if (event == cv::EVENT_MBUTTONUP)
    {

    }
    else if (event == cv::EVENT_MBUTTONDOWN)
    {

    }

}

void debugWindow::updateDisplayImage()
{

    if (!imageProcessing.empty())
        imageProcessing.release();

    imageProcessing = image1.clone();

    if (selectBlobIndex != -1)
    {
        // P.S.: �����drawContours �]��Harvie�t��k��X�Ӫ�Contour���O�s��,�O�H���ƦC, ��drawContours �e�X�ӷ|�ܩ_�� 
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

            std::cout << "--------��ܹ϶��S�x----------" << endl;
            std::cout << "���n          :" + to_string(vBlobList[selectBlobIndex].Area()) << endl;
            std::cout << "���e��        :" + to_string(vBlobList[selectBlobIndex].AspectRatio()) << endl;
            std::cout << "Circularity   :" + to_string(vBlobList[selectBlobIndex].Circularity()) << endl;
            std::cout << "Rectangularity:" + to_string(vBlobList[selectBlobIndex].Rectangularity()) << endl;
            std::cout << "Bulkiness     :" + to_string(vBlobList[selectBlobIndex].Bulkiness()) << endl;
            std::cout << "Compactness   :" + to_string(vBlobList[selectBlobIndex].Compactness()) << endl;
            std::cout << "���b          :" + to_string(vBlobList[selectBlobIndex].Ra()) << endl;
            std::cout << "�u�b          :" + to_string(vBlobList[selectBlobIndex].Rb()) << endl;
            std::cout << "Roundness     :" + to_string(vBlobList[selectBlobIndex].Roundness()) << endl;
            std::cout << "Sides         :" + to_string(vBlobList[selectBlobIndex].Sides()) << endl;
        }

    }
    else
    {
        if (isNeedRefresh)
        {
            system("cls");
            std::cout << "�Цb�����W�I��ƹ��k���� �w�Q���Ҫ��϶�" << endl;
        }
    }

    cv::resize(imageProcessing, displayImage, cv::Size(), zoomFactor, zoomFactor);
    cv::imshow(strTitle, displayImage);
    isNeedRefresh = false;
}

void debugWindow::doKeyAction(int key)
{
    if (key == '1')
        selectImageIndex = 1;
    else if (key == '2')
        selectImageIndex = 2;

}

void debugWindow::Run(Mat imageInput, vector<BlobInfo> BlobInfo)
{
    system("cls");

    vBlobList = BlobInfo;

    image1 = imageInput.clone();
    cv::namedWindow(strTitle);
    cv::setMouseCallback("��ܹϤ�", debugWindow::onMouseStatic, &image1);
    roiRect = Rect(0,0, image1.cols,image1.rows);

    while (true) {
        updateDisplayImage();
        int key = cv::waitKey(30);
        if (key == 27) break; // ���UESC��h�X

        doKeyAction(key);
        //if (key == '+') scale *= 1.1; // ��j
        //if (key == '-') scale /= 1.1; // �Y�p
        //if (key == 'w') offset.y -= 500; // �W��
        //if (key == 's') offset.y += 500; // �U��
        //if (key == 'a') offset.x -= 500; // ����
        //if (key == 'd') offset.x += 500; // �k��
    }
}

#pragma endregion


void ShowDebugWindow(Mat Img_Ref, vector<BlobInfo> BlobInfo)
{
    debugWindow dbW = debugWindow();
    dbW.Run(Img_Ref, BlobInfo);
    dbW.~debugWindow();
}


