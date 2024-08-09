#pragma once
#include "OpenCV_Extension_Tool.h"

#include <iostream>
#include <fstream>
using namespace std;

void readFileParameter(string path, bool& isReadFileSucceed, vector<float>& fParamArr);
void saveFileParameter(string path, bool& isReadFileSucceed, vector<float>  fParamArr);

void ShowDebugWindow(Mat Img_Ref,vector<BlobInfo> BlobInfo);

void GetAllFolderImage(string FolderPath,vector<string>& vStr);