//
// Created by true on 19-3-25.
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

int main()
{
    string strSettingPath = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/MonoBSS/Monocular/TUM3.yaml";
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_32F);
    ORBExtractorPara *ORBExtractorParameter = new ORBExtractorPara;
    ReadFromYaml(strSettingPath, K, DistCoef, ORBExtractorParameter);
    cout << "K : " << K << endl;

    cv::Mat ImgRgb1 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/imageByself/imageDB_1.jpg");
    cv::Mat ImgRgb2 = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/imageByself/imageDB_3.jpg");
    cv::Mat mImGray1 ;
    cv::Mat mImGray2 ;
    //图像转化为灰度图;
    cv::cvtColor(ImgRgb1, mImGray1, CV_RGB2GRAY);
    cv::cvtColor(ImgRgb2, mImGray2, CV_RGB2GRAY);
    cv::imshow("mInGray1 : ",mImGray1);
    //建立特征提取器
    ORB_SLAM2::ORBextractor *mpORBextractor = new ORBextractor(3 * ORBExtractorParameter->nFeatures, ORBExtractorParameter->fScaleFactor, ORBExtractorParameter->nLevels, ORBExtractorParameter->fIniThFAST,
                                                               ORBExtractorParameter->fMinThFAST);

    //构造Frame类,主要是真的方便呀--  唯一在我们实验中不需要构造的是三个数据.即ORBVocabulary,bf,thDepth.后者均可以设0,词典提取.
    ORB_SLAM2::ORBVocabulary *mpORBVocabulary = new ORBVocabulary();
    float bf = 0.0;
    float thDepth = 0.0;
    ORB_SLAM2::Frame mCurrentFrame1 = ORB_SLAM2::Frame(mImGray1, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf,
                                                       thDepth);
    ORB_SLAM2::Frame mCurrentFrame2 = ORB_SLAM2::Frame(mImGray2, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef, bf,
                                                       thDepth);
    cout << "Frame1.mvKeys.size():" << mCurrentFrame1.mvKeysUn.size() << endl;

    //获取两帧图像的位姿;并解算特征点的3D位置;
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

    int nmatches = matcher.SearchForInitialization(mCurrentFrame1, mCurrentFrame2, mvbPrevMatched, mvIniMatches, 200);
    cout << "nmatches" << "   " << nmatches << endl;

    //先进行RANSAC,位姿解算;
    vector<cv::KeyPoint> keypoints1;  //keypoints 存储了被匹配上的所有特征点;
    vector<cv::KeyPoint> keypoints2;
    /*typedef pair<int, int> Match;
    vector<Match> mvMatches12;       //mvMatch存储了 在第一帧与第二针匹配特征点各自的index.*/
    //int N=mvIniMatches.size();
    //cout<<" N "<<N<<endl;


    int countnumber=0;
    for (int i = 0; i < mvIniMatches.size(); i++) {
        if (mvIniMatches[i] >= 0) {
            //cv::DMatch currentMatch;
            cv::KeyPoint points1;
            cv::KeyPoint points2;
            points1=mCurrentFrame1.mvKeysUn[i];
            points2=mCurrentFrame2.mvKeysUn[mvIniMatches[i]];
            //currentMatch.queryIdx =i;
            //currentMatch.trainIdx =i;
            //matches.push_back(currentMatch);  match本来存储所有被成功匹配的点的坐标. 目前不需要了.
            keypoints1.push_back(points1);
            keypoints2.push_back(points2);  //keypoints1和keypoints2之间匹配的次序是okay的；
            countnumber++;

        } else
            continue;
    }

    //观察两张图的匹配问题;
    int width = ImgRgb1.cols;
    int height = ImgRgb1.rows;
    cv::Size wholeSize(width * 2, height);
    cv::Mat outImg(wholeSize, ImgRgb1.type());;
    DrawMatches(keypoints1, keypoints2, ImgRgb1, ImgRgb2, outImg);
    cv::imshow("MatchesImg :", outImg);

    cout<<keypoints1.size()<<endl;
    cout<<"coutnumber : "<<countnumber<<endl;  //匹配上的特征点数量；

    //建立静态特征点集合,是过去的判断，在track过程中进行更新，补充，删减，重建,是track过程中对连续帧定位解算的关键；
    vector<cv::KeyPoint> StaticInformationInFrames;
    StaticInformationInFrames=keypoints1;


    //重新计算F矩阵并验证投影误差值;
    const int N = keypoints1.size();

    // Indices for minimum set selection
    // 新建一个容器vAllIndices，生成0到N-1的数作为特征点的索引
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    // 步骤2：在所有匹配特征点对中随机选择8对匹配特征点为一组，共选择mMaxIterations组
    // 用于FindHomography和FindFundamental求解

    int mMaxIterations=2000;
    vector<vector<size_t> > mvSets;
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            // 产生0到N-1的随机数
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            // idx表示哪一个索引对应的特征点被选中
            int idx = vAvailableIndices[randi];
            //cout<<"idx : "<<idx<<endl;
            mvSets[it][j] = idx;

            // randi对应的索引已经被选过了，从容器中删除
            // randi对应的索引用最后一个元素替换，并删掉最后一个元素
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    //利用RANSAC计算F21矩阵;
    // 归一化;
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(keypoints1,vPn1, T1);
    Normalize(keypoints2,vPn2, T2);
    cv::Mat T2t = T2.t();

    //迭代变量
    cv::Mat F21;
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);  //这样模型内点在所有特征点的位置中有两个vector<bool>来描述,一个是所有特征中的序列,一个是在匹配上的特征点的序列位置;
    float currentScore;

    // 最佳结果变量;其中vbMatchesInliers获取最佳8点;
    vector<bool>vbMatchesInliers(N,false);
    float score = 0.0;

    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[idx];
            vPn2i[j] = vPn2[idx];
        }

        //利用8点解算相机位姿;
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
        F21i = T2t*Fn*T1;

        currentScore=CheckFundamental(F21i,keypoints1,keypoints2,vbCurrentInliers);
        //cout<<"here is still right"<<currentScore<<endl;
        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers=vbCurrentInliers;
            score = currentScore;
        }
    }

    int numberGoodPoint=0;
    for(int i=0;i<vbMatchesInliers.size();i++)
    {
        if(vbMatchesInliers[i]){
            cv::Point2f p1, p2;
            p1.x = keypoints1[i].pt.x;
            p1.y = keypoints1[i].pt.y;
            p2.x = keypoints2[i].pt.x;
            p2.y = keypoints2[i].pt.y;
            circle(ImgRgb1, p1, 4, cv::Scalar(100, 150, 255));
            numberGoodPoint++;
        }
    }
    cv::imshow("StaticPoint in Img1: ", ImgRgb1);
    cout<<"numberGoodPoint : "<<numberGoodPoint<<endl; //模型内点的个数；


   // cout<<"有这么多的好特征点,看算出来RT怎么样把"<<numberGoodPoint<<endl;
    cv::Mat R,t;
    //调整fundamental的数据类型,使之与K矩阵保持一致;
    //int Ktype = K.type();
    //int Ftype =fundamental_matrix.type();
    //fundamental_matrix.convertTo(fundamental_matrix,Ktype);

    vector<bool>vbGood=vector<bool>(keypoints1.size(),false);
    //保存部分三维点坐标；
    cv::Mat Zeros31 = cv::Mat::zeros(3, 1, CV_32F);
    vector<cv::Mat>p3dC1List=vector<cv::Mat>(keypoints1.size(),Zeros31);
    //直接在该步骤中获得R,T,3D点,以及图像1中的特征点坐标及其与3D点的对应关系.啊`
    if(ComputeRTForF(R,t,F21,keypoints1,keypoints2,vbMatchesInliers,K))
    {
        cout<<"R"<<R<<endl;
        cout<<"t"<<t<<endl;

        //准备参数,用于三角化;
        // Calibration parameters
        const float fx = K.at<float>(0,0);
        const float fy = K.at<float>(1,1);
        const float cx = K.at<float>(0,2);
        const float cy = K.at<float>(1,2);

        //构建P1和P2矩阵,同时计算相机光心,光心用以计算3D点的视差;
        // Camera 1 Projection Matrix K[I|0]
        // 步骤1：得到一个相机的投影矩阵
        // 以第一个相机的光心作为世界坐标系
        cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
        K.copyTo(P1.rowRange(0,3).colRange(0,3));
        // 第一个相机的光心在世界坐标系下的坐标
        cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

        // Camera 2 Projection Matrix K[R|t]
        // 步骤2：得到第二个相机的投影矩阵
        cv::Mat P2(3,4,CV_32F);
        R.copyTo(P2.rowRange(0,3).colRange(0,3));
        t.copyTo(P2.rowRange(0,3).col(3));
        P2 = K*P2;
        // 第二个相机的光心在世界坐标系下的坐标
        cv::Mat O2 = -R.t()*t;

        vector<float>vCosParallax=vector<float>(keypoints1.size(),0.0); // 用于记录成功三角化的点的视差;

        cv::Point3f Zero3d;
        Zero3d.x=0;Zero3d.y=0;Zero3d.z=0;
        vector<cv::Point3f> vP3D=vector<cv::Point3f>(keypoints1.size(),Zero3d);
        float th2=3.874;

        //进行三角化,并记录成功三角化的特征点index
        for(int i=0;i<keypoints1.size();i++)
        {
            //先获取特征点;
            if(vbMatchesInliers[i]){
            cv::KeyPoint kp1;
            cv::KeyPoint kp2;
            kp1=keypoints1[i];
            kp2=keypoints2[i];

            cv::Mat p3dC1;

            // 步骤3：利用三角法恢复三维点p3dC1
            Triangulate(kp1,kp2,P1,P2,p3dC1);
            //cout<<p3dC1<<endl;

            //输出3D点坐标
            //cout<<"3D坐标x:  "<<p3dC1.at<float>(0);
            //cout<<"3D坐标y:  "<<p3dC1.at<float>(1);
            //cout<<"3D坐标z:  "<<p3dC1.at<float>(2)<<endl;
            //判断是否有无穷大值;
            if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
            {
                vbGood[i]=false;
                continue;
            }
            //cout<<"坐标中没有无穷大值"<<endl;
            //计算视差;
            cv::Mat normal1 = p3dC1 - O1;
            float dist1 = cv::norm(normal1);
            cv::Mat normal2 = p3dC1 - O2;
            float dist2 = cv::norm(normal2);
            float cosParallax = normal1.dot(normal2)/(dist1*dist2);
            //cout<<"cosParallax  : "<<cosParallax<<endl;

            //若3D点的值为负值;
            if(p3dC1.at<float>(2)<=0 && cosParallax<1)  //初始为0.99998;0.999995对应10度;
            {
                vbGood[i]=false;
                continue;
            }

            cv::Mat p3dC2 = R*p3dC1+t;
            if(p3dC2.at<float>(2)<=0 && cosParallax<1) //初始为0.99998;0.999995对应10度;
            {
                vbGood[i]=false;
                continue;
            }

            //计算重投影误差,如果误差过大,则淘汰;
            float im1x, im1y;
            float invZ1 = 1.0/p3dC1.at<float>(2);
            im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
            im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
            float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
            //cout<<"squareError1  :  "<<squareError1<<endl;

            if(squareError1>th2)
            {
                vbGood[i]=false;
                continue;
            }
            //cout<<"当前3D点误差为:  "<<squareError1<<endl;
            // Check reprojection error in second image
            // 计算3D点在第二个图像上的投影误差
            float im2x, im2y;
            float invZ2 = 1.0/p3dC2.at<float>(2);
            im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
            im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

            float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);
            //cout<<"squareError2  :  "<<squareError2<<endl;
            // 步骤6.2：重投影误差太大，跳过淘汰
            if(squareError2>th2)
            {
                vbGood[i]=false;
                continue;
            }

            //保存视差,保存3D点坐标; 次序为keypoints的次序,不满足三角化要求的点均置0;

            vCosParallax[i]=cosParallax;
            vP3D[i] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));

            if(cosParallax<1)
                vbGood[i]=true;
            if(vbGood[i])
                p3dC1List[i]=p3dC1;
            }
        }

        int Success3D=0;
        for(int i=0;i<vbGood.size();i++){
            if(vbGood[i])
            {
                Success3D++;
            }
        }
        cout<<"成功三角化的点个数 :   "<<Success3D<<endl;
    };

     //显示被成功三角化的特征点;
     vector<cv::KeyPoint>CurrentModelInliers;
    for(int i=0;i<vbGood.size();i++)
    {
        if(vbGood[i]){
            cv::Point2f p1, p2;
            p1.x = keypoints1[i].pt.x;
            p1.y = keypoints1[i].pt.y;
            p2.x = keypoints2[i].pt.x;
            p2.y = keypoints2[i].pt.y;
            CurrentModelInliers.push_back(keypoints1[i]);
            circle(ImgRgb1, p1, 4, cv::Scalar(200, 0, 255));

        }
    }
    cv::imshow("StaticPoint in Img1: ", ImgRgb1);

    //进行判断，静态特征集合中是否出现了运动物体并对定位解算产生了影响；
    //判断依据：1.匹配上的静态特征集合的分布度与模型解算的模型分布之比；
    //判断依据: 2.可三角化的点数除以匹配上的特征点点数；

    //首先计算匹配上的点数的特征点分布；在本case中为keypoints1的分布度；

    float StaticDistirbution;
    StaticDistirbution=DistributionCalculate(StaticInformationInFrames);
    float CurrentModelDistribution;
    CurrentModelDistribution=DistributionCalculate(CurrentModelInliers);
    float DistributionAttenuation=0.0;
    DistributionAttenuation=CurrentModelDistribution/StaticDistirbution;
    cout<<"Distribution Attenuation : "<<DistributionAttenuation<<endl;

    //计算成功三角化的点的个数除以静态特征点
    float ModelInliersRatop=(float)CurrentModelInliers.size()/keypoints1.size();
    cout<<"ModelInliersRatop : "<<ModelInliersRatop<<endl;


    float threshold;
    // 当点数下降过多时，分布度的变化会发生不稳定，由于可能保留了那些较为边缘的点，导致分布度反而增加；
    // 在分布度下降的较多的时候，我们需要重新审视我们所采用的
    // 可三角化点的数目
    //  if()



    cv::waitKey(0);
    return 0;
}
