#pragma once
#include "OpenCV_Extension_Tool.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <io.h>
#include <direct.h>

using namespace std;

void readFileParameter(string path, bool& isReadFileSucceed, vector<float>& fParamArr);
void saveFileParameter(string path, bool& isReadFileSucceed, vector<float>  fParamArr);

void ShowDebugWindow(Mat Img_Ref,vector<BlobInfo> BlobInfo);

void GetAllFolderBmpImage(string FolderPath,vector<string>& vStr);