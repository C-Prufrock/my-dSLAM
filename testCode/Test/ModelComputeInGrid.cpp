//
// Created by true on 18-12-14.
//

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

int main() {
    //获取相机参数;
    string strSettingPath = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/MonoBSS/Monocular/TUM3.yaml";
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_32F);
    ORBExtractorPara *ORBExtractorParameter = new ORBExtractorPara;
    ReadFromYaml(strSettingPath, K, DistCoef, ORBExtractorParameter);


    //建立循环读取图像的函数
    //首先获取图像的id;
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();

    //for(int ni =0;ni<nImages;ni++) {
    // 接收图像,提取特征
    int ni =43;
    cv::Mat ImgRgb1 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/"+vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
    //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[17] << endl;
    cv::Mat ImgRgb2 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/"+vstrImageFilenames[ni+3], CV_LOAD_IMAGE_UNCHANGED);
    //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[18] << endl;
    cv::Mat mImGray1;
    cv::Mat mImGray2;
    //图像转化为灰度图;
    cv::cvtColor(ImgRgb1, mImGray1, CV_RGB2GRAY);
    cv::cvtColor(ImgRgb2, mImGray2, CV_RGB2GRAY);
    cout<<"Size of ImageRgb1: "<< ImgRgb1.size()<<endl;

    //建立特征提取器
    ORB_SLAM2::ORBextractor *mpORBextractor = new ORBextractor(4*ORBExtractorParameter->nFeatures, ORBExtractorParameter->fScaleFactor, ORBExtractorParameter->nLevels, ORBExtractorParameter->fIniThFAST, ORBExtractorParameter->fMinThFAST);

    //构造Frame类,主要是真的方便呀--  唯一在我们实验中不需要构造的是三个数据.即ORBVocabulary,bf,thDepth.后者均可以设0,词典提取.
    ORB_SLAM2::ORBVocabulary *mpORBVocabulary = new ORBVocabulary();
    float bf = 0.0;
    float thDepth = 0.0;
    ORB_SLAM2::Frame mCurrentFrame1 = ORB_SLAM2::Frame(mImGray1, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf, thDepth);
    ORB_SLAM2::Frame mCurrentFrame2 = ORB_SLAM2::Frame(mImGray2, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf, thDepth);
    cout << "Frame1.mvKeys.size():" << mCurrentFrame1.mvKeysUn.size() << endl;

    //初始化 初始器;
    Initializer *mpInitializer = new Initializer(mCurrentFrame1, 1.0, 200);

    cout << mpInitializer->GetmvKeys1Size() << endl;

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
    ////画出格子;
    //每个bin的大小为bsize;//格子的大小 事实上,划分的方式都将会对图像的解算有巨大的影响.
    int bsizex = 120;
    int bsizey = 120;
    int nGridrows = ImgRgb1.rows / bsizey;
    cout << "nGridrows : " << nGridrows << endl;
    int nGridcols = ImgRgb1.cols / bsizex;
    cout << "nGridcols:  " << nGridcols << endl;

    for (int i = 0; i < nGridcols + 1; i++)   //画格子竖线;
    {
        cv::Point2f p1;
        cv::Point2f p2;
        p1.y = 0;
        p1.x = bsizex * i;
        p2.y = nGridrows * bsizey;
        p2.x = bsizex * i;
        for (int j = 0; j < nGridrows + 1; j++) {
            cv::Point2f p3;
            cv::Point2f p4;
            p3.x = 0;
            p3.y = bsizey * i;
            p4.x = nGridcols * bsizex;
            p4.y = bsizey * i;
            cv::line(ImgRgb1, p1, p2, cv::Scalar(255, 0, 0));   ///在ImgRgb1上画竖线;
            cv::line(ImgRgb1, p3, p4, cv::Scalar(255, 0, 0));   ///在ImgRgb1上画横线;
        }
    }


    //cv::imshow("ImgRgb1 : ",FeatureImg);
    //cv::waitKey(0);

    ////*****/////
    //获得被成功匹配的特征点;
    ///*****/////
    vector<cv::KeyPoint> keypoints1;  //keypoints 存储了被匹配上的所有特征点;
    vector<cv::KeyPoint> keypoints2;
    typedef pair<int, int> Match;
    vector<Match> mvMatches12;       //mvMatch存储了 在第一帧与第二针匹配特征点各自的index.

    for (int i = 0; i < mvIniMatches.size(); i++) {
        if (mvIniMatches[i] >= 0) {
            //cv::DMatch currentMatch;
            Match currentMatchindex;
            //currentMatch.queryIdx =i;
            //currentMatch.trainIdx =i;
            currentMatchindex.first = i;
            currentMatchindex.second = mvIniMatches[i];
            //matches.push_back(currentMatch);  match本来存储所有被成功匹配的点的坐标. 目前不需要了.
            keypoints1.push_back(mCurrentFrame1.mvKeysUn[i]);
            keypoints2.push_back(mCurrentFrame2.mvKeysUn[mvIniMatches[i]]);
            mvMatches12.push_back(currentMatchindex);
        } else
            continue;
    }
    //cout<<keypoints1.size()<<endl;

    cv::Mat FeatureImg;
    cv::drawKeypoints(ImgRgb1, keypoints1, FeatureImg, cv::Scalar(255, 0, 255), 0); //画格子中的特征点;
    cv::imshow("FeatureIMg :", FeatureImg);

    //获得每个方格子中的特征点,如果点数过少, .--实际没有这么做;
    //cout<<"Is there no idea?"<<mvIniMatches.size()<<endl;
    vector<int> mGrid[nGridcols][nGridrows];
    //初始化mGrid;
    //将匹配的特征点分配到Grid[i][j];
    //cout<<1/3<<endl;
    for(int i=0;i<mvIniMatches.size();i++)  //利用匹配较好的点.
    {
        if(mvIniMatches[i]>=0){
            int posX=mCurrentFrame1.mvKeysUn[i].pt.x/bsizex;
            int posY=mCurrentFrame1.mvKeysUn[i].pt.y/bsizey;
            if(posX<0 || posX>=nGridcols || posY<0 || posY>=nGridrows)
            {
                continue;
            }else{
                mGrid[posX][posY].push_back(i);  //mGrid保存了第一帧中可找到匹配点的特征点的index.
            }
        }
    }

    //记录mGrid中 有特征点的格子;
    vector<vector<int>>GridHas;
    for(int i=0;i<nGridcols;i++)
    {
        for(int j=0;j<nGridrows;j++)
        {
            if(mGrid[i][j].size()>8){
                GridHas.push_back(mGrid[i][j]);
                //cout<<"GridHas"<<GridHas<<endl;
                //cout<< "当前列"<<i<<endl;
                //cout<<"当前行"<<j<<endl;
            }
            //for(int k=0;k<mGrid[i][j].size();k++)
            //cout<<mGrid[i][j][k]<<endl;
        }
    }
    cout<<"特征点数量大于8的格子: "<<GridHas.size()<<endl;

    ////对每个格子进行随机采样,保留含有匹配上特征点的格子.并对格子进行随机采样.
    //随机取8个格子;
    ///取一个格子的8个点 计算模型,首先验证自身模型是否正确(原因是svd分解,匹配误差以及离散采样导致的    误差),
    /// 多遍历几次,每一次遍历计算外点得分数,得分数过高,说明该格子的模型混乱度过高,故舍去.
    ///为得到该阈值,我们计算格子中的采样模型,记录每个模型的外点分数之和.
    /// 如正确并在全局范围内,检验模型的适应度.
    ///首先我们可以观察一下情况 - - -实际把主要的代码都码了
    //vector<cv::Mat> Modellist;
    vector<vector<bool>> vbMatchInliersList;


    //for(int iGrid=0;iGrid<GridHas.size();iGrid++) {
        int iGrid=0;
        vector<size_t> vAllIndices;
        vAllIndices.reserve(GridHas[iGrid].size());   //vAllIndices的长度设置为包含当前格子包含的特征点个数;
        cout << "feautures numbers of GridHas[iGrid]: " << GridHas[iGrid].size() << endl;
        vector<size_t> vAvailableIndices;

        for (int i = 0; i < GridHas[iGrid].size(); i++) {
            vAllIndices.push_back(i);
        }
        cout<<"features number in GridHas[Grid]"<<vAllIndices.size()<<endl;
        ///根据GridHas
        int nMaxIterations = 2;
        vector<vector<size_t> > mvSets;
        mvSets = vector<vector<size_t> >(nMaxIterations, vector<size_t>(8, 0));
        DUtils::Random::SeedRandOnce(0);

        for (int it = 0; it < nMaxIterations; it++) //循环次数;
        {
            vAvailableIndices = vAllIndices;
            for (size_t j = 0; j < 8; j++) {

                ///产生1-格子总数(含特征点)的随机数;
                //int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
                //对格子中的特征点进行随机选取.
                int randfInGrid = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1); //在格子中随机选取一特征点.
                //cout<<"randi : "<<randi<<endl;
                //cout<<"randfInGrid"<<randfInGrid<<endl;
                mvSets[it][j] = vAvailableIndices[randfInGrid];   ///包含的值为匹配上的特征点在CurrentFrame1.upKeys的index.

                //cout<<mvSets[it][j]<<endl;

                vAvailableIndices[randfInGrid] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }
        }
        //}*/


        /**
         * @brief 计算基础矩阵
         *
         * 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分 //加上分布的呦!!!!
        */
       
        ///归一化坐标系;
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        //归一化;
        Normalize(mCurrentFrame1.mvKeysUn, vPn1, T1);
        Normalize(mCurrentFrame2.mvKeysUn, vPn2, T2);
        cv::Mat T2t = T2.t();

        //保留最优结果.
        float score = 0.0;
        vector<bool> vbMatchesInliers = vector<bool>(mCurrentFrame1.mvKeysUn.size(), false); //保留符合模型的特征点;
        vector<bool> vbMatchesInliersInAllFeatures = vector<bool>(mCurrentFrame1.mvKeysUn.size(), false);

        // 迭代中的变量;
        vector<cv::Point2f> vPn1i(8);     //随机取的8个点 在1帧中;
        vector<cv::Point2f> vPn2i(8);     //随机取的8个点 在2帧中;

        float currentScore = 0;               //当前模型的分数;
        //vector<int>index(8,0);      //最优模型的8个内点是什么;
        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        //const float th =3.841;
        //const float thScore = 5.991;  //5.991实际为H模型的判定阈值;
        const float sigma = 1.0;
        const float invSigmaSquare = 1.0 / (sigma * sigma);
        vector<cv::KeyPoint> KeypointInliers;   //用于存储属于运动模型内点的特征点,用于计算其分布.
        int bestidx = 0;
        cv::Mat F21i;
        cv::Mat F21 = cv::Mat::zeros(3, 3, CV_32F);                 //F21i是随机取的点所解算的模型;
        //根据随机选取的8个点,计算F模型;
        for (int it = 0; it < nMaxIterations; it++) {
            //选中最小集合;
            vector<bool> vbCurrentInliers(mCurrentFrame1.mvKeysUn.size(), false);   //当前ransac解算运动模型包含的内点;
            vector<bool> vbCurrentInliersInAllFeatures(mCurrentFrame1.mvKeysUn.size(), false);   //当前ransac解算运动模型包含的内点,以格子中数目为其大小;
            //获取8个点.

            for (int j = 0; j < 8; j++) {
                int idx = mvSets[it][j];
                vPn1i[j] = vPn1[idx];
                vPn2i[j] = vPn2[mvIniMatches[idx]];
            }

            cv::Mat Fn = ComputeF21(vPn1i, vPn2i);
            F21i = T2t * Fn * T1;
            //minSetScore.resize(8);
            //计算8个最小集合与模型的拟合度如何,如果过差则舍弃.
            if (checkMaxminSets(F21i, vPn1i, vPn2i)) {
                //currentScore= CheckFundamentalBasedOnDistribution(F21i,F21,nmatches,keypoints1,keypoints2,3.87,sigma,vbCurrentInliers,KeypointInliers,it);
                //currentScore=CheckFundamentalBasedOnModel(F21i,F21,nmatches,keypoints1,keypoints2,3.84,sigma,vbCurrentInliers,KeypointInliers,it);
                currentScore = CheckDominatFundamental(F21i, mvIniMatches, mCurrentFrame1.mvKeysUn, GridHas[iGrid],
                                                       mCurrentFrame2.mvKeysUn, 3.871, sigma, vbCurrentInliers,
                                                       vbCurrentInliersInAllFeatures, it);
            } else {
                continue;
            }
            /*
             * 保留最大分数,进入下一步循环.
             */
            /*if (currentScore > score) {
                F21 = F21i;
                vbMatchesInliers = vbCurrentInliers;
                vbMatchesInliersInAllFeatures = vbCurrentInliersInAllFeatures;
                score = currentScore;
                bestidx = it;
                //MaxminSetScore=minSetScore;
            }
        }*/
        /*cout << "格子中解算得到的最优F模型为: " << endl << F21 << endl;
        cout << "success here for solve F" << endl;*/

    cv::waitKey(0);
    return 0;
}
}

