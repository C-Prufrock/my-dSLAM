//
// Created by true on 18-12-13.
//

#include "Tracking.h"  //Tracking.h中包含了Map.h Frame.h等等头文件,放心用数据.
#include "toolsForTest.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBextractor.h"
#include "Frame.h"
#include "ORBmatcher.h"
#include "Initializer.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

using namespace std;
using namespace ORB_SLAM2;

#include<thread>

int main() {
    //获取相机参数;
    string strSettingPath = "/home/fzj/VslamBasedStaticDescribeSet/MonoBSS/Monocular/TUM1.yaml";
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_32F);
    ORBExtractorPara *ORBExtractorParameter = new ORBExtractorPara;
    ReadFromYaml(strSettingPath, K, DistCoef, ORBExtractorParameter);

    //建立循环读取图像的函数
    //首先获取图像的id;
    /*
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();*/


    //for(int ni =0;ni<nImages;ni++) {
        //接收图像,提取特征
        //int ni=43;
        cv::Mat ImgRgb1 = cv::imread("/home/fzj/VslamBasedStaticDescribeSet/Vocabulary/imageByself/imageDB_1.jpg", CV_LOAD_IMAGE_UNCHANGED);
        //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[17] << endl;
        cv::Mat ImgRgb2 = cv::imread("/home/fzj/VslamBasedStaticDescribeSet/Vocabulary/imageByself/imageDB_3.jpg", CV_LOAD_IMAGE_UNCHANGED);
        //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[18] << endl;
        cv::Mat mImGray1;
        cv::Mat mImGray2;
        //图像转化为灰度图;
        cv::cvtColor(ImgRgb1, mImGray1, CV_RGB2GRAY);
        cv::cvtColor(ImgRgb2, mImGray2, CV_RGB2GRAY);
        cout << "Size of ImageRgb1: " << ImgRgb1.size() << endl;

        //建立特征提取器
        ORB_SLAM2::ORBextractor *mpORBextractor = new ORBextractor(3 * ORBExtractorParameter->nFeatures, ORBExtractorParameter->fScaleFactor, ORBExtractorParameter->nLevels, ORBExtractorParameter->fIniThFAST, ORBExtractorParameter->fMinThFAST);

        //构造Frame类,主要是真的方便呀--  唯一在我们实验中不需要构造的是三个数据.即ORBVocabulary,bf,thDepth.后者均可以设0,词典提取.
        ORB_SLAM2::ORBVocabulary *mpORBVocabulary = new ORBVocabulary();
        float bf = 0.0;
        float thDepth = 0.0;
        ORB_SLAM2::Frame mCurrentFrame1 = ORB_SLAM2::Frame(mImGray1, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf, thDepth);
        ORB_SLAM2::Frame mCurrentFrame2 = ORB_SLAM2::Frame(mImGray2, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf, thDepth);

        /*画出当前图像的特征点*/
        /*
        cv::Mat img1,img2;
        cv::drawKeypoints(mImGray1, mCurrentFrame1.mvKeysUn, img1, cv::Scalar(255, 0, 255), 0);
        cv::drawKeypoints(mImGray2, mCurrentFrame2.mvKeysUn, img2, cv::Scalar(255, 0, 255), 0);
        cv::imshow("img1: ", img1);
        cv::imshow("img2: ",img2);*/
        //cv::waitKey(0);
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

        //if( nmatches<100)   //如果匹配的点数不足100,就别用了,匹配度这么低 匹配也不一定稳定.
            //continue;
        //else
        //cout<<"current index of Image(匹配数目大于100哇!!!!) :" << ni<<endl;
        cout << "nmatches" << "   " << nmatches << endl;


        ////画出格子;
        //每个bin的大小为bsize;
        int bsize = 180;
        int nGridrows = ImgRgb1.rows / bsize;
        //cout << "nGridrows : " << nGridrows << endl;
        int nGridcols = ImgRgb1.cols / bsize;
        //cout << "nGridcols:  " << nGridcols << endl;

        for (int i = 0; i < nGridcols + 1; i++)   //画格子竖线;
        {
            cv::Point2f p1;
            cv::Point2f p2;
            p1.y = 0;
            p1.x = bsize * i;
            p2.y = nGridrows * bsize;
            p2.x = bsize * i;
            for (int j = 0; j < nGridrows + 1; j++) {
                cv::Point2f p3;
                cv::Point2f p4;
                p3.x = 0;
                p3.y = bsize * i;
                p4.x = nGridcols * bsize;
                p4.y = bsize * i;
                cv::line(ImgRgb1, p1, p2, cv::Scalar(255, 0, 0));   ///在ImgRgb1上画线;
                cv::line(ImgRgb1, p3, p4, cv::Scalar(255, 0, 0));
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

        cv::Mat FeatureImg;
        cv::drawKeypoints(ImgRgb1, keypoints1, FeatureImg, cv::Scalar(255, 0, 255), 0); //画格子中的特征点;
        cv::imshow("FeatureIMg :", FeatureImg);
        cout<<"keypoints.size() : "<<keypoints1.size()<<endl;
        //cout<<keypoints1.size()<<endl;

        ///该段代码用于ransac迭代.
        int nMaxIterations = 8000;
        vector<vector<size_t> > mvSets;
        mvSets = vector<vector<size_t> >(nMaxIterations, vector<size_t>(8, 0));
        DUtils::Random::SeedRandOnce(0);

        vector<size_t> vAllIndices;
        vAllIndices.reserve(nmatches);   //vAllIndices保留格子的index;
        vector<size_t> vAvailableIndices;

        for (int i = 0; i < nmatches; i++) {
            vAllIndices.push_back(i);
        }
        ////迭代,随机取点;
        for (int it = 0; it < nMaxIterations; it++) {
            vAvailableIndices = vAllIndices;
            // Select a minimum set
            for (size_t j = 0; j < 8; j++) {
                // 产生0到N-1的随机数
                int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
                //cout<<"random number : "<<randi<<endl;
                // idx表示哪一个索引对应的特征点被选中
                int idx = vAvailableIndices[randi];

                mvSets[it][j] = idx;  //获得第it次迭代  第一帧中第j个随机特征点;

                // randi对应的索引已经被选过了，从容器中删除
                // randi对应的索引用最后一个元素替换，并删掉最后一个元素
                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }
        }
        /**
         * @brief 计算基础矩阵
         *
         * 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分 //加上分布的呦!!!!
        */
        ///归一化坐标系;
        vector<cv::Point2f> vPn1, vPn2; //vPn1 与vPn2记录了ransac选中的8个特征点;
        //vector<int>indexBest(8,0);      //最优模型的8个内点是什么;
        cv::Mat T1, T2;
        Normalize(keypoints1, vPn1, T1);
        Normalize(keypoints2, vPn2, T2);
        cv::Mat T2t = T2.t();

        //保留最优结果.
        float score = 0.0;
        vector<bool> vbMatchesInliers = vector<bool>(nmatches, false); //保留符合模型的特征点;

        // 迭代中的变量;
        vector<cv::Point2f> vPn1i(8);     //随机取的8个点 在1帧中;
        vector<cv::Point2f> vPn2i(8);     //随机取的8个点 在2帧中;
        cv::Mat F21i, F21;                 //F21i是随机取的点所解算的模型;
        vector<bool> vbCurrentInliers(nmatches, false);   //当前ransac解算运动模型包含的内点;
        float currentScore;               //当前模型的分数;
        //vector<int>index(8,0);      //最优模型的8个内点是什么;

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        const float th = 3.481;
        const float thScore = 5.991;

        const float sigma = 1.0;
        const float invSigmaSquare = 1.0 / (sigma * sigma);
        vector<cv::KeyPoint> KeypointInliers;   //用于存储属于运动模型内点的特征点,用于计算其分布.
        int bestidx;
        vector<int>Best8pointsModel;
        //执行 加分布计算的ransac 并保留最优结果
        for (int it = 0; it < nMaxIterations; it++) {
            //选中最小集合;
            vector<int>Ini8Points;
            for (int j = 0; j < 8; j++) {
                int idx = mvSets[it][j];
                //cout<<"idx(看随机采样的点是否重复) :"<<idx<<endl;
                vPn1i[j] = vPn1[idx];
                vPn2i[j] = vPn2[idx];
                Ini8Points.push_back(idx);
                //cout<<vPn1i[j]<<endl;
                //cout<<vPn2i[j]<<endl;
                //index[j]=idx;
            }
            cv::Mat Fn = ComputeF21(vPn1i, vPn2i);
            F21i = T2t * Fn * T1;

            //currentScore= CheckFundamentalBasedOnDistribution(F21i,F21,nmatches,keypoints1,keypoints2,3.841,1,vbCurrentInliers,KeypointInliers,it);
            currentScore = CheckFundamentalBasedOnModel(F21i, F21, nmatches, keypoints1, keypoints2, 3.84, 1,vbCurrentInliers, KeypointInliers,it);
            /*
             * 保留最大分数,进入下一步循环.
             */

            if (currentScore > score) {
                F21 = F21i;
                vbMatchesInliers = vbCurrentInliers;
                Best8pointsModel=Ini8Points;
                score = currentScore;
                bestidx=it;
            }
        }
        cout<<"bestidx   " <<bestidx<<endl;
        cout<<"score :   "<<score<<endl;
        /// 画出解算的模型内点,或者说静态点,表为红色. 在ImgRgb1 上进行画图,该图已经画了格子;
        int inliersnumber = 0;
        for (int i = 0; i < nmatches; i++) {
            if (vbMatchesInliers[i]) {
                cv::Point2f p1;

                p1.x = keypoints1[i].pt.x;
                p1.y = keypoints1[i].pt.y;
                cout<<"p1.x: "<<p1.x<<endl;
                cout<<"p1.y: "<<p1.y<<endl;
                circle(ImgRgb1, p1, 4, cv::Scalar(0, 255, 0));
                inliersnumber++;
            }
        }
       for(int i=0;i<Best8pointsModel.size();i++)
       {
        cv::Point2f p1;
        p1.x = keypoints1[Best8pointsModel[i]].pt.x;
        p1.y = keypoints1[Best8pointsModel[i]].pt.y;
        circle(ImgRgb1, p1, 4, cv::Scalar(0, 0, 255));
       }
        cout << "finally,inliersnumber: " << inliersnumber << endl;
        if(inliersnumber>65){

            //cout <<"current index of image: "<<ni<<endl;
            //break;
        }
        cv::imshow("ImgRgb1 inlers:", ImgRgb1);
        cv::imwrite("/home/fzj/VslamBasedStaticDescribeSet/Result/Ransac8000.jpg",ImgRgb1);
        cv::waitKey(0);
    //}


    return 0;
}
