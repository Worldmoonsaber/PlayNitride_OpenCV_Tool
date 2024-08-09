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
