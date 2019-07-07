//
/* Created by true on 18-12-5.
** Author: Xiaoyun Lu
 * @brief
 *
 * 采用光流法进行初始化,由于要区分动态物体与静态物体,因此要允许探测large displacement的光流.
 * large displacement 可能是由于非刚体的运动产生,比如人的胳膊与头部的warp
 * 关键是对光流进行聚类,如何进一步保证聚类的结果尽可能的不在静态区域内有动态物体.
 *
*/

#include "Tracking.h"  //Tracking.h中包含了Map.h Frame.h等等头文件,放心用数据.

#include<iostream>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Optimizer.h"
#include"PnPsolver.h"

#include "ORBextractor.h"
#include "Frame.h"
#include "ORBmatcher.h"

using namespace std;
using namespace ORB_SLAM2;

class ORBExtractorPara
{

public:
    ORBExtractorPara(){};

    int nFeatures;
    float fScaleFactor;
    int nLevels;
    int fIniThFAST;
    int fMinThFAST;
};
void ReadFromYaml(string& strSettingPath,cv::Mat& K,cv::Mat& DistCoef,ORBExtractorPara* ORBExtractorParameter)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];


    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;


    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }

    ORBExtractorParameter->nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    ORBExtractorParameter->fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    ORBExtractorParameter->nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    ORBExtractorParameter->fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    ORBExtractorParameter->fMinThFAST = fSettings["ORBextractor.minThFAST"];

};

void DrawMatches(const vector<int>mvIniMatches,Frame mCurrentFrame1,Frame mCurrentFrame2,cv::Mat ImgRgb1,cv::Mat ImgRgb2,cv::Mat& outImg,int& nmatches)
{
    vector<cv::DMatch>matches;
    vector<cv::KeyPoint>keypoints1;
    keypoints1.resize(nmatches);
    vector<cv::KeyPoint>keypoints2;
    keypoints2.resize(nmatches);
    for(int i=0;i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]>=0){
            cv::DMatch currentMatch;
            currentMatch.queryIdx =i;
            currentMatch.trainIdx =i;
            matches.push_back(currentMatch);
            keypoints1.push_back(mCurrentFrame1.mvKeysUn[i]);
            keypoints2.push_back(mCurrentFrame2.mvKeysUn[mvIniMatches[i]]);

        }
        else
            continue;
    }
    //构造outImg
    int width=ImgRgb1.cols;
    int height=ImgRgb1.rows;
    cv::Mat roi1=outImg(cv::Rect(0,0,width,height));
    ImgRgb1.copyTo(roi1);


    //画出特征点并连接二者
    for(int i=0;i<=nmatches*2;i++)
    {
        cv::Point2f p1;
        p1.x=keypoints1[i].pt.x;
        p1.y=keypoints1[i].pt.y;

        cv::Point2f p2;
        p2.x=keypoints2[i].pt.x;
        p2.y=keypoints2[i].pt.y;


        cv::circle(outImg,p1,3,cv::Scalar(0, 0, 255));
        cv::circle(outImg,p2,3,cv::Scalar(0, 255, 0));
        cv::line(outImg,p1,p2,cv::Scalar(255,0,0));
    }
    //画出所有的特征点.
    //cv::Scalar matchColor=cv::Scalar::all(-1);
    //cv::Scalar singlePointColor=cv::Scalar::all(255);
    //cv::drawMatches(ImgRgb1,mCurrentFrame1.mvKeysUn,ImgRgb2,mCurrentFrame2.mvKeysUn,matches,outImg);
};

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);
int main()
{
    //获取相机参数;
    string strSettingPath="/home/true/CLionProjects/VslamBasedStaticDescribeSet/MonoBSS/Monocular/TUM1.yaml";
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    cv::Mat DistCoef=cv::Mat(4,1,CV_32F);
    ORBExtractorPara* ORBExtractorParameter = new ORBExtractorPara;
    ReadFromYaml(strSettingPath,K,DistCoef,ORBExtractorParameter);

    //建立循环读取图像的函数
    //首先获取图像的id;
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();

    for(int ni=0;ni<nImages;ni+=2)
    {
        //接收图像,提取特征
        cv::Mat ImgRgb1 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat ImgRgb2 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/"+vstrImageFilenames[ni+3],CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat mImGray1;
        cv::Mat mImGray2;
        //图像转化为灰度图;
        cv::cvtColor(ImgRgb1, mImGray1, CV_RGB2GRAY);
        cv::cvtColor(ImgRgb2, mImGray2, CV_RGB2GRAY);

        //建立特征提取器
        ORB_SLAM2::ORBextractor *mpORBextractor = new ORBextractor(2*ORBExtractorParameter->nFeatures,ORBExtractorParameter->fScaleFactor, ORBExtractorParameter->nLevels, ORBExtractorParameter->fIniThFAST, ORBExtractorParameter->fMinThFAST);
        ORB_SLAM2::ORBextractor *mpORBextractor2 = new ORBextractor(2*ORBExtractorParameter->nFeatures,ORBExtractorParameter->fScaleFactor, ORBExtractorParameter->nLevels, ORBExtractorParameter->fIniThFAST, ORBExtractorParameter->fMinThFAST);

        //构造Frame类,主要是真的方便呀--  唯一在我们实验中不需要构造的是三个数据.即ORBVocabulary,bf,thDepth.后者均可以设0,词典提取.
        ORB_SLAM2::ORBVocabulary *mpORBVocabulary = new ORBVocabulary();
        float bf = 0.0;
        float thDepth = 0.0;
        ORB_SLAM2::Frame mCurrentFrame1 = ORB_SLAM2::Frame(mImGray1, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf, thDepth);
        ORB_SLAM2::Frame mCurrentFrame2 = ORB_SLAM2::Frame(mImGray2, 0.0, mpORBextractor2, mpORBVocabulary, K, DistCoef, bf, thDepth);

        //画图--第一副图像的特征点
        cv::Mat img;
        cv::drawKeypoints(mImGray1, mCurrentFrame1.mvKeysUn, img, cv::Scalar(255, 0, 255), 0);
        //cv::imshow("img: ", img);
        cout << "Frame1.mvKeys.size():" << mCurrentFrame1.mvKeysUn.size() << endl;

        //进行初始化;
        Initializer *mpInitializer = new Initializer(mCurrentFrame1, 1.0, 200);

        //进行匹配;
        ORB_SLAM2::ORBmatcher matcher(0.9, true);
        std::vector<int> mvIniMatches;             // 跟踪初始化时前两帧匹配,第二帧特征点的index值.不在此处用-1的原因,是因为-1是无符号整型.
        fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        std::vector<cv::Point2f> mvbPrevMatched;   // 记录匹配的2d坐标.
        mvbPrevMatched.resize(mCurrentFrame1.mvKeysUn.size());
        for(size_t i=0; i<mCurrentFrame1.mvKeysUn.size(); i++)
            mvbPrevMatched[i]=mCurrentFrame1.mvKeysUn[i].pt;

        if(mCurrentFrame2.mvKeys.size()<=100)
        {
            cout<< "the keyspoints is not enough in currentframe! "<<endl;
        }
        else {
            int nmatches = matcher.SearchForInitialization(mCurrentFrame1, mCurrentFrame2, mvbPrevMatched, mvIniMatches,50);
            cout << "nmatches"<<"   " <<nmatches << endl;
            //获得满足匹配条件的匹配点,并画出匹配条件.

            int width=ImgRgb1.cols;
            int height=ImgRgb1.rows;
            cv::Size wholeSize(width,height);
            cv::Mat outImg(wholeSize,ImgRgb1.type());;
            DrawMatches(mvIniMatches, mCurrentFrame1, mCurrentFrame2, ImgRgb1, ImgRgb2, outImg,nmatches);
            cv::imshow("matched img: ", outImg);
            cv::waitKey(0);


            //vector<cv::Point2f>MatchedSatisfy;
            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated(nmatches,-1); // Triangulated Correspondences (mvIniMatches)
            std::vector<cv::Point3f> mvIniP3D;
            mpInitializer->Initialize(mCurrentFrame2, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated);

            // 删除那些无法进行三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }
            cout<<"nmatches after poseCaculate : " << "  " <<  nmatches << endl;



            //DrawMatches(mvIniMatches, mCurrentFrame1, mCurrentFrame2, ImgRgb1, ImgRgb2, outImg,nmatches);


        }

        /*if (nmatches>=100)
        {
            cout<<"nmathes: "<< nmatches<<endl;
            cv::waitKey(0);
            break;
        }*/

    }
    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;  //把s给ss;
            double t;
            string sRGB;
            ss >> t;  //把ss给t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

