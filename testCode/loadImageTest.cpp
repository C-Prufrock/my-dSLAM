//
// Created by true on 18-11-29.
//
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include <typeinfo>

using namespace std;

int main()
{
    string strFile="/home/true/CLionProjects/VslamBasedStaticDescribeSet/Thirdparty/rgbd_dataset_freiburg1_xyz/rgb.txt";
    cout<<strFile<<endl;
    vector<string>vstrImageFilenames;
    vector<double> vTimestamps;

    ifstream f;
    f.open(strFile.c_str());


    // skip first three lines
    string s0;
    getline(f,s0);  //getline函数 返回下一行;
    cout<<s0<<endl;
    getline(f,s0);
    cout<<s0<<endl;
    getline(f,s0);
    cout<<s0<<endl;

    while(!f.eof())
    {
        string s;
        getline(f,s);
        double a=0.0;
        //cout<<s<<endl;
        //cout<<typeid(s).name()<<endl;

        if(!s.empty())
        {
            stringstream ss;
            ss << s;  //把s给ss;
            double t;
            string sRGB;
            ss >> t;  //把ss给t;
            vTimestamps.push_back(t);
            //cout<<"Timgestamps: "<<vTimestamps[0]<<endl;
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
            //cout<<"imagesequence: "<<vstrImageFilenames[0]<<endl;
        }
    }

    return 0;

}

