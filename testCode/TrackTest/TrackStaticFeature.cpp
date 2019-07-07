//
// Created by true on 18-12-25.
/*
 *
 *思路brief
 * 1.建立位姿解算特征点集合.
 * 2.初始化 并三角化
 * 3.在特征点几何分布合理,且点数较多时,或者当前符合运动模型的特征点较少时
 * 4.进行特征点集合的补充与更新
 * 5.验证最终结果.
 */
//
#include "Tracking.h"  //Tracking.h中包含了Map.h Frame.h等等头文件,放心用数据.
#include "Converter.h"
#include "toolsForTest.h"
#include<iostream>
#include<fstream>
#include<chrono>
#include<math.h>
#include<string>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "ORBextractor.h"
#include "Frame.h"
#include "ORBmatcher.h"
#include "Initializer.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

using namespace std;
using namespace ORB_SLAM2;
using namespace cv;


int main(){

    string strSettingPath = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/MonoBSS/Monocular/TUM3.yaml";
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_32F);
    ORBExtractorPara *ORBExtractorParameter = new ORBExtractorPara;
    ReadFromYaml(strSettingPath, K, DistCoef, ORBExtractorParameter);
    cout << "K : " << K << endl;

    //建立循环读取图像的函数
    //首先获取图像的id,文件名;
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    string goal;
    goal="rgb/1341846669.118346.png";
    int nImages = vstrImageFilenames.size();
    for(int ni =0;ni<nImages;ni++) {
        //接收图像,提取特征
        //int ni = 43;
        cv::Mat ImgRgb1 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/"+vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        cout<<vstrImageFilenames[ni]<<endl;
        cv::Mat ImgRgb2 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/"+vstrImageFilenames[ni+1], CV_LOAD_IMAGE_UNCHANGED);
        //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[18] << endl;
        string b=vstrImageFilenames[ni];
        if(b.compare(goal)==0)
        {
            cout<<"current ni"<<ni<<endl;
            break;
        }
        cv::Mat mImGray1;
        cv::Mat mImGray2;
        //图像转化为灰度图;

        cv::cvtColor(ImgRgb1, mImGray1, CV_RGB2GRAY);
        cout << "Size of ImageRgb2: " << ImgRgb2.size() << endl;
        cv::cvtColor(ImgRgb2, mImGray2, CV_RGB2GRAY);


        imshow("current image :",ImgRgb1);
        waitKey(20);
    }



    return 0;
}