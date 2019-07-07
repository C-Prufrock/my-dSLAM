#include "Tracking.h"  //Tracking.h中包含了Map.h Frame.h等等头文件,放心用数据.
#include "Converter.h"
#include "toolsForTest.h"
#include<iostream>
#include<fstream>
#include<chrono>
#include<math.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "ORBextractor.h"
#include "Frame.h"
#include "ORBmatcher.h"
#include "Initializer.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

using namespace std;
using namespace ORB_SLAM2;

Eigen::Vector3d NormalizeEuler_angles(Eigen::Vector3d euler_angles);  //对轴角进行归一化;
float MOdelDiffBetweenGrid(cv::Mat Vector3d1,cv::Mat Vector3d2,cv::Mat t1,cv::Mat t2)
{
    float sum1=0;
    float sum2=0;
    for(int i=0;i<3;i++)
    {
        sum1+=pow(fabs(Vector3d1.at<float>(0,i)-Vector3d2.at<float>(0,i)),2);
    }
    for(int j=0;j<3;j++)
    {
        sum2+=pow(fabs(t1.at<float>(0,j)-t2.at<float>(0,j)),2);
    }
    return (sum1+sum2);
};
int main() {
    //获取相机参数;
    string strSettingPath = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/MonoBSS/Monocular/TUM3.yaml";
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_32F);
    ORBExtractorPara *ORBExtractorParameter = new ORBExtractorPara;
    ReadFromYaml(strSettingPath, K, DistCoef, ORBExtractorParameter);
    cout<<"K : "<<K<<endl;

    //建立循环读取图像的函数
    //首先获取图像的id,文件名;
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();

    //for(int ni =0;ni<nImages;ni++) {
    // 接收图像,提取特征
    int ni =2;
    cv::Mat ImgRgb1 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/"+vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
    //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[17] << endl;
    cv::Mat ImgRgb2 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/"+vstrImageFilenames[ni+6], CV_LOAD_IMAGE_UNCHANGED);
    //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[18] << endl;
    cv::Mat mImGray1;
    cv::Mat mImGray2;
    //图像转化为灰度图;
    cv::cvtColor(ImgRgb1, mImGray1, CV_RGB2GRAY);
    cv::cvtColor(ImgRgb2, mImGray2, CV_RGB2GRAY);
    cout<<"Size of ImageRgb1: "<< ImgRgb1.size()<<endl;

    //建立特征提取器
    ORB_SLAM2::ORBextractor *mpORBextractor = new ORBextractor(3.5*ORBExtractorParameter->nFeatures, ORBExtractorParameter->fScaleFactor, ORBExtractorParameter->nLevels, ORBExtractorParameter->fIniThFAST, ORBExtractorParameter->fMinThFAST);

    //构造Frame类,主要是真的方便呀--  唯一在我们实验中不需要构造的是三个数据.即ORBVocabulary,bf,thDepth.后者均可以设0,词典提取.
    ORB_SLAM2::ORBVocabulary *mpORBVocabulary = new ORBVocabulary();
    float bf = 0.0;
    float thDepth = 0.0;
    ORB_SLAM2::Frame mCurrentFrame1 = ORB_SLAM2::Frame(mImGray1, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf, thDepth);
    ORB_SLAM2::Frame mCurrentFrame2 = ORB_SLAM2::Frame(mImGray2, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf, thDepth);
    cout << "Frame1.mvKeys.size():" << mCurrentFrame1.mvKeysUn.size() << endl;

    //初始化 初始器;
    Initializer *mpInitializer = new Initializer(mCurrentFrame1, 1.0, 200);

    //进行匹配;
    ORB_SLAM2::ORBmatcher matcher(0.9, true);
    std::vector<int> mvIniMatches;             // 该变量包含参考帧中所有特征点对应的匹配点.-1表示无法匹配上.跟踪初始化时前两帧匹配,第二帧特征点的index值.不在此处用-1的原因,是因为-1是无符号整型.
    fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
    std::vector<cv::Point2f> mvbPrevMatched;   // 记录匹配的2d坐标.
    mvbPrevMatched.resize(mCurrentFrame1.mvKeysUn.size());
    for (size_t i = 0; i < mCurrentFrame1.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame1.mvKeysUn[i].pt;

    int nmatches = matcher.SearchForInitialization(mCurrentFrame1, mCurrentFrame2, mvbPrevMatched, mvIniMatches, 100);
    cout << "nmatches" << "   " << nmatches << endl;

    ////画出格子;
    //每个bin的大小为bsize;
    int bsize=120;
    int nGridrows=ImgRgb1.rows/bsize;cout<<"nGridrows : "<<nGridrows<<endl;
    int nGridcols=ImgRgb1.cols/bsize;cout<<"nGridcols:  "<<nGridcols<<endl;

    for(int i=0;i<nGridcols+1;i++)   //画格子竖线;
    {
        cv::Point2f p1;
        cv::Point2f p2;
        p1.y=0;
        p1.x=bsize*i;
        p2.y=nGridrows*bsize;
        p2.x=bsize*i;
        for(int j=0;j<nGridrows+1;j++){
            cv::Point2f p3;
            cv::Point2f p4;
            p3.x=0;
            p3.y=bsize*i;
            p4.x=nGridcols*bsize;
            p4.y=bsize*i;
            cv::line(ImgRgb1,p1,p2,cv::Scalar(255,0,0));   ///在ImgRgb1上画线;
            cv::line(ImgRgb1,p3,p4,cv::Scalar(255,0,0));
        }
    }

    //cv::imshow("ImgRgb1 : ",FeatureImg);
    //cv::waitKey(0);

    ////*****/////
    //获得被成功匹配的特征点;
    ///*****/////

    cout<<"success here!!"<<endl;
    //初始化mGrid;
    //将匹配的特征点分配到Grid[i][j];
    vector<int> mGrid[nGridcols][nGridrows];

    for(int i=0;i<mvIniMatches.size();i++)  //利用匹配较好的点.
    {
        if(mvIniMatches[i]>=0){
            int posX=mCurrentFrame1.mvKeysUn[i].pt.x/bsize;
            int posY=mCurrentFrame1.mvKeysUn[i].pt.y/bsize;
            if(posX<0 || posX>=nGridcols || posY<0 || posY>=nGridrows)
            {
                continue;
            }else{
                mGrid[posX][posY].push_back(i);  //mGrid保存了第一帧中可找到匹配点的特征点的index.
            }
        }
    }

    //记录mGrid中 有特征点的格子;实际上是大于8个特征点的格子.
    vector<vector<int>>GridHas;
    for(int i=0;i<nGridcols;i++)
    {
        for(int j=0;j<nGridrows;j++)
        {
            if(mGrid[i][j].size()>8){
                GridHas.push_back(mGrid[i][j]);
                //cout<< "当前列"<<i<<endl;
                //cout<<"当前行"<<j<<endl;
            }
            //for(int k=0;k<mGrid[i][j].size();k++)
            //cout<<mGrid[i][j][k]<<endl;
        }
    }
    cout<<"含有特征点的数量: "<<GridHas.size()<<endl;

    ////对每个格子进行随机采样,保留含有匹配上特征点的格子.并对格子进行随机采样.
    //随机取8个格子;

    ///取一个格子的8个点 计算模型,首先验证自身模型是否正确(原因是svd分解,匹配误差以及离散采样导致的误差),
    /// 多遍历几次,每一次遍历计算外点得分数,得分数过高,说明该格子e1,mCurrentFrame2,GridMatches12);
    //的模型混乱度过高,故舍去.
    ///为得到该阈值,我们计算格子中的采样模型,记录每个模型的外点分数之和.
    /// 如正确并在全局范围内,检验模型的适应度.
    ///首先我们可以观察一下情况 - - -实际把主要的代码都码了
    vector<cv::Mat> F21ilist;
    vector<vector<bool>> vbMatchInliersList;
    //for(int iGrid=0;iGrid<GridHas.size();iGrid++) {
    int iGrid=2;
    vector<size_t> vAllIndices;
    vAllIndices.reserve(GridHas[iGrid].size());   //vAllIndices保留格子的index;
    cout << "feautures numbers of GridHas[iGrid]: " << GridHas[iGrid].size() << endl;
    cv::Mat R,t,F;

    vector<cv::KeyPoint> keypoints1;  //keypoints 存储了被匹配上的所有特征点;
    vector<cv::KeyPoint> keypoints2;
    vector<cv::DMatch>GridMatches12;

    //构造格子中的匹配特征点的matches.
    for(int i=0;i<GridHas[iGrid].size();i++)
    {
        cv::DMatch Matches;
        Matches.queryIdx=GridHas[iGrid][i];
        Matches.trainIdx=mvIniMatches[GridHas[iGrid][i]];
        GridMatches12.push_back(Matches);
        keypoints1.push_back(mCurrentFrame1.mvKeysUn[GridHas[iGrid][i]]);
        keypoints2.push_back(mCurrentFrame1.mvKeysUn[mvIniMatches[GridHas[iGrid][i]]]);
    }
    cout<<"keypoints1 number :"<<keypoints1.size()<<endl;
    cout<<"keypoints2 number :"<<keypoints2.size()<<endl;
    cout<<"GridMatches12 number"<<GridMatches12.size()<<endl;

    pose_estimation_2d2d(keypoints1,keypoints2,GridMatches12,F);
    cout<<F<<endl;

    cv::waitKey(0);
    return 0;
}

Eigen::Vector3d NormalizeEuler_angles(Eigen::Vector3d euler_angles){
    float sum=0;
    for(int i=0;i<3;i++)
    {
        sum+=fabs(euler_angles(i,0));
    }
    for(int j=0;j<3;j++)
    {
        euler_angles(j,0)=euler_angles(j,0)/sum;
    }
    return euler_angles;
};


