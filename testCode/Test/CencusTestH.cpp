//
// Created by true on 18-12-19.
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

Eigen::Vector3d NormalizeEuler_angles(Eigen::Vector3d euler_angles);  //对轴角进行归一化;
float MOdelDiffBetweenGrid(cv::Mat Vector3d1,cv::Mat Vector3d2,cv::Mat t1,cv::Mat t2)
{
    float sum11=0;
    //float sum12=0;
    float sum21=0;

    for(int i=0;i<3;i++) {
        sum11 += fabs(Vector3d1.at<float>(0, i)) - fabs(Vector3d2.at<float>(0, i));
    }
    //cout<<"sum11 : "<<"  "<<sum11<<endl;

    for(int j=0;j<3;j++)
    {
        sum21+=fabs(fabs(t1.at<float>(0,j))-fabs(t2.at<float>(0,j)));

    }
    //cout<<"sum21 : "<<"  "<<sum21<<endl;
    return(sum11+sum21);

};
int main() {
    //获取相机参数;
    string strSettingPath = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/MonoBSS/Monocular/TUM1.yaml";
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_32F);
    ORBExtractorPara *ORBExtractorParameter = new ORBExtractorPara;
    ReadFromYaml(strSettingPath, K, DistCoef, ORBExtractorParameter);
    cout << "K : " << K << endl;

    //建立循环读取图像的函数
    //首先获取图像的id,文件名;
    /*用来读取图像序列.
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = "/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();*/

    //for(int ni =0;ni<nImages;ni++) {
    // 接收图像,提取特征
    //int ni = 43;
    cv::Mat ImgRgb1 = cv::imread("/home/true/MYNT-EYE-D-SDK/samples/_output/bin/1.jpg", CV_LOAD_IMAGE_UNCHANGED);
    //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[17] << endl;
    cv::Mat ImgRgb2 = cv::imread("/home/true/MYNT-EYE-D-SDK/samples/_output/bin/3.jpg", CV_LOAD_IMAGE_UNCHANGED);
    //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[18] << endl;
    cv::Mat mImGray1;
    cv::Mat mImGray2;
    //图像转化为灰度图;
    cv::cvtColor(ImgRgb1, mImGray1, CV_RGB2GRAY);
    cv::cvtColor(ImgRgb2, mImGray2, CV_RGB2GRAY);
    cout << "Size of ImageRgb1: " << ImgRgb1.size() << endl;

    //建立特征提取器
    ORB_SLAM2::ORBextractor *mpORBextractor = new ORBextractor(3 * ORBExtractorParameter->nFeatures,
                                                               ORBExtractorParameter->fScaleFactor,
                                                               ORBExtractorParameter->nLevels,
                                                               ORBExtractorParameter->fIniThFAST,
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

    int nmatches = matcher.SearchForInitialization(mCurrentFrame1, mCurrentFrame2, mvbPrevMatched, mvIniMatches, 300);
    cout << "nmatches" << "   " << nmatches << endl;

    ////画出格子;
    //每个bin的大小为bsize;
    int bsize = 240;
    int nGridrows = ImgRgb1.rows / bsize;
    cout << "nGridrows : " << nGridrows << endl;
    int nGridcols = ImgRgb1.cols / bsize;
    cout << "nGridcols:  " << nGridcols << endl;

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
    //cout<<keypoints1.size()<<endl;
    cv::Mat FeatureImg;
    cv::drawKeypoints(ImgRgb1, keypoints1, FeatureImg, cv::Scalar(255, 0, 255), 0); //画格子中的特征点;
    cv::imshow("FeatureIMg :", FeatureImg);

    //用于显示两张匹配的图像;
    int width = ImgRgb1.cols;
    int height = ImgRgb1.rows;
    cv::Size wholeSize(width * 2, height);
    cv::Mat outImg(wholeSize, ImgRgb1.type());;
    DrawMatches(keypoints1, keypoints2, ImgRgb1, ImgRgb2, outImg);
    cv::imshow("MatchesImg :", outImg);

    //初始化mGrid;
    //将匹配的特征点分配到Grid[i][j];
    vector<int> mGrid[nGridcols][nGridrows];

    for (int i = 0; i < mvIniMatches.size(); i++)  //利用匹配较好的点.
    {
        if (mvIniMatches[i] >= 0) {
            int posX = mCurrentFrame1.mvKeysUn[i].pt.x / bsize;
            int posY = mCurrentFrame1.mvKeysUn[i].pt.y / bsize;
            if (posX < 0 || posX >= nGridcols || posY < 0 || posY >= nGridrows) {
                continue;
            } else {
                mGrid[posX][posY].push_back(i);  //mGrid保存了第一帧中可找到匹配点的特征点的index.
            }
        }
    }

    //记录mGrid中 有特征点的格子;实际上是大于8个特征点的格子.
    vector<vector<int>> GridHas;
    for (int i = 0; i < nGridcols; i++) {
        for (int j = 0; j < nGridrows; j++) {
            if (mGrid[i][j].size() > 8) {
                GridHas.push_back(mGrid[i][j]);
                //cout<< "当前列"<<i<<endl;
                //cout<<"当前行"<<j<<endl;
            }
            //for(int k=0;k<mGrid[i][j].size();k++)
            //cout<<mGrid[i][j][k]<<endl;
        }
    }
    cout << "包含特征点数大于8的格子数目: " << GridHas.size() << endl;

    ///取一个格子的8个点 计算模型,首先验证自身模型是否正确(原因是svd分解,匹配误差以及离散采样导致的误差),
    /// 多遍历几次,每一次遍历计算外点得分数,得分数过高,说明该格子的模型混乱度过高,故舍去.
    ///为得到该阈值,我们计算格子中的采样模型,记录每个模型的外点分数之和.
    /// 如正确并在全局范围内,检验模型的适应度.
    ///首先我们可以观察一下情况 - - -实际把主要的代码都码了


    //保留得到的F模型 R,T以及模型的内点;
    vector<cv::Mat> F21ilist;
    vector<vector<bool>> vbMatchInliersList;
    vector<cv::Mat> RList;
    vector<cv::Mat> TList;
    vector<int> GoodGrid;


    //主循环,遍历每一个格子;
    for(int iGrid=0;iGrid<GridHas.size();iGrid++) {
    //int iGrid = 2;
    /**
     * @brief 计算基础矩阵
     *
     * 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分 //加上分布的呦!!!!
    */
    ///归一化坐标系;
    vector<cv::Point2f> vPn1, vPn2; //vPn1 与vPn2记录了格子归一化后的特征点
    cv::Mat T1, T2;
    vector<cv::KeyPoint> keypoints1InGrids;  //keypoints 格子中存储了被匹配上的所有特征点;
    vector<cv::KeyPoint> keypoints2InGrids;
    //归一化; 只根据格子中的特征点进行归一化.
    for (int i = 0; i < GridHas[iGrid].size(); i++) {
        //cout << "i :" << i << endl;
        keypoints1InGrids.push_back(mCurrentFrame1.mvKeysUn[GridHas[iGrid][i]]);
        //cout << "keypointsInGrids1 position: " << keypointsInGrids1[i].pt.x << "   " << keypointsInGrids1[i].pt.y<< endl;
        keypoints2InGrids.push_back(mCurrentFrame2.mvKeysUn[mvIniMatches[GridHas[iGrid][i]]]);
        //cout << "keypointsInGrids2 position: " << keypointsInGrids2[i].pt.x << "   " << keypointsInGrids2[i].pt.y<< endl;
    }
    Normalize(keypoints1InGrids, vPn1, T1);
    Normalize(keypoints2InGrids, vPn2, T2);
    cv::Mat T2t = T2.inv();
    //cout << "归一化T1矩阵: " << endl << T1 << endl;
    //cout << "归一化T2矩阵: " << endl << T2 << endl;


    //为随机取点做准备;
    vector<size_t> vAllIndices;
    vAllIndices.reserve(vPn1.size());   //vAllIndices的长度设置为包含当前格子包含的特征点个数;
    cout << "feautures numbers of GridHas[iGrid]: " << GridHas[iGrid].size() << endl;
    vector<size_t> vAvailableIndices;

    for (int i = 0; i < vPn1.size(); i++) {
        vAllIndices.push_back(i); //vAllIndices保存的值为当前格子中特征点的index;
    }

    ///根据GridHas
    int nMaxIterations = 200;
    vector<vector<size_t> > mvSets;
    mvSets = vector<vector<size_t> >(nMaxIterations, vector<size_t>(8, 0));
    DUtils::Random::SeedRandOnce(0);

    for (int it = 0; it < nMaxIterations; it++) //循环次数;
    {
        vAvailableIndices = vAllIndices;
        for (size_t j = 0; j < 8; j++)    //随机取格子中的8个特征点
        {
            //对格子中的特征点进行随机选取.
            //cout<<"vAvailableIndices.size() :"<<"  "<<vAvailableIndices.size()<<endl;
            int randfInGrid = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1); //在格子中随机选取一特征点.
            mvSets[it][j] = vAvailableIndices[randfInGrid];   ///包含的值为格子中特征点的排序index
            //cout<<"mvSets[it][j] :"<<"  "<<mvSets[it][j]<<endl;
            vAvailableIndices[randfInGrid] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    //保留最优结果.
    float score = 0.0;
    float currentScore = 0;               //当前模型的分数;
    vector<bool> vbMatchesInliers = vector<bool>(vPn1.size(), false);    //保留符合模型的特征点;
    vector<bool> vbMatchesInliersInAllFeatures = vector<bool>(mCurrentFrame1.mvKeysUn.size(), false);
    // 迭代中的变量;
    vector<cv::Point2f> vPn1i(8);     //随机取的8个点 在1帧中;
    vector<cv::Point2f> vPn2i(8);     //随机取的8个点 在2帧中;
    cv::Mat H21i = cv::Mat::zeros(3, 3, CV_32F);
    cv::Mat H12i = cv::Mat::zeros(3, 3, CV_32F);
    cv::Mat H21 = cv::Mat::zeros(3, 3, CV_32F);                 //F21i是随机取的点所解算的模型,F21是获得的得分最高的模型;

    //vector<int>index(8,0);      //最优模型的8个内点是什么;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    //const float th =3.841;
    //const float thScore = 5.991;
    const float sigma = 1.0;
    const float invSigmaSquare = 1.0 / (sigma * sigma);
    //vector<cv::KeyPoint> KeypointInliers;   //用于存储属于运动模型内点的特征点,用于计算其分布.
    /**
     *
         typedef pair<float, float> fMatch;
         vector<fMatch>minSetScore;
         vector<fMatch>MaxminSetScore;
     */
    int bestidx = 0;
    //执行 加分布计算的ransac 并保留最优结果
    for (int it = 0; it < nMaxIterations; it++) {
        //选中最小集合;
        vector<bool> vbCurrentInliers(vPn1.size(), false);   //当前ransac解算运动模型包含的内点,以格子中数目为其大小;
        vector<bool> vbCurrentInliersInAllFeatures(mCurrentFrame1.mvKeysUn.size(), false);   //当前ransac解算运动模型包含的内点,以格子中数目为其大小;
        //获取8个点.
        for (int j = 0; j < 8; j++) {
            int idx = mvSets[it][j];
            //cout<<"idx : "<<idx<<endl;
            vPn1i[j] = vPn1[idx];
            vPn2i[j] = vPn2[idx];
            //cout<<"vPn1i[j]"<<vPn1i[j]<<endl;
            //cout<<"vPn2i[j]"<<vPn2i[j]<<endl;
        }

        cv::Mat Hn = ComputeH21(vPn1i, vPn2i);
        H21i = T2t * Hn * T1;
        //cout<<"H21: "<<"  "<<H21i<<endl;
        H12i = H21i.inv();
        //minSetScore.resize(8);
        //计算8个最小集合与模型的拟合度如何,如果过差则舍弃.

        //currentScore= CheckFundamentalBasedOnDistribution(F21i,F21,nmatches,keypoints1,keypoints2,3.87,sigma,vbCurrentInliers,KeypointInliers,it);
        //currentScore=CheckFundamentalBasedOnModel(F21i,F21,nmatches,keypoints1,keypoints2,3.84,sigma,vbCurrentInliers,KeypointInliers,it);
        //currentScore = CheckDominatFundamental(F21i, mvIniMatches, mCurrentFrame1.mvKeysUn, GridHas[iGrid],mCurrentFrame2.mvKeysUn, 3.871, sigma, vbCurrentInliers,vbCurrentInliersInAllFeatures,it);
        currentScore = CheckDominatHomograph(H21i, H12i, mvIniMatches, mCurrentFrame1.mvKeysUn, GridHas[iGrid],
                                             mCurrentFrame2.mvKeysUn, 3.871, sigma, vbCurrentInliers,
                                             vbCurrentInliersInAllFeatures,
                                             it);
        //cout << currentScore << endl;
        /*
         * 保留最大分数,进入下一步循环.
         */
        if (currentScore > score) {
            H21 = H21i;
            //cout<<"H21 : "<<"  "<<endl<<H21<<endl;
            vbMatchesInliers = vbCurrentInliers;
            vbMatchesInliersInAllFeatures = vbCurrentInliersInAllFeatures;
            score = currentScore;
            bestidx = it;
            //MaxminSetScore=minSetScore;
        }
    }
    //cout << "得到的最优H21: " << endl << H21 << endl;

    /// 画出解算的模型内点,或者说静态点,表为红色. 在ImgRgb1 上进行画图,该图已经画了格子;
    //利用所有内点 最小二乘重新计算F矩阵
    vector<cv::Point2f> vPnInliers1;
    vector<cv::Point2f> vPnInliers2;
    int inliersnumber = 0;
    for (int i = 0; i < vbMatchesInliers.size(); i++) {
        if (vbMatchesInliers[i]) {
            vPnInliers1.push_back(vPn1[i]);
            //cout<<"vPn1[i]"<<vPn1[i]<<endl;
            vPnInliers2.push_back(vPn2[i]);
            //cout<<"vPn2[i]"<<vPn2[i]<<endl;
            inliersnumber++;
        }
    }

    int inliersnumber2 = 0;

    for (int i = 0; i < vbMatchesInliersInAllFeatures.size(); i++) {
        if (vbMatchesInliersInAllFeatures[i]) {
            cv::Point2f p1;
            p1.x = mCurrentFrame1.mvKeysUn[i].pt.x;
            p1.y = mCurrentFrame1.mvKeysUn[i].pt.y;
            circle(ImgRgb1, p1, 4, cv::Scalar(0, 255, 255));
            inliersnumber2++;
        }
    }
    //cout<<" vbMatchesInliersInAllFeatures.size()"<<" "<< vbMatchesInliersInAllFeatures.size()<<endl<<"inliersnumber2 :"<<"  "<<inliersnumber2<<endl;
    cv::imshow("ImgRgb1 inlers:", ImgRgb1);

    //cout << "vPnInliers1 numbers:" << vPnInliers1.size() << endl;
    //cout << "vPnInliers2 numbers:" << vPnInliers2.size() << endl;
    //cout << "final,inliersnumber: " << inliersnumber << endl;
    //利用vPnInliers1/2重新计算F矩阵

    cv::Mat Hm = ComputeH21(vPnInliers1, vPnInliers2);
    cout << "Hm with all inliers :" << endl << Hm << endl;
    cv::Mat H21Final = T2t * Hm * T1;
    //cout << "格子中解算得到的F模型为: " << endl << F21Final << endl;


    cv::imshow("ImgRgb1 inlers:", ImgRgb1);
    //cout << "score" << score << endl;
    cout << "currend index of iGrid" << "  " << iGrid << endl;

    cv::Mat R, t;

    if (ComputeRTForH(R, t, H21Final, mCurrentFrame1.mvKeysUn, mCurrentFrame2.mvKeysUn, vbMatchesInliersInAllFeatures,
                      K, mvIniMatches)) {
        Eigen::Matrix<double, 3, 3> Rmatrix = Converter::toMatrix3d(R);  //将RList转换为Eigen类型;
        Eigen::Vector3d euler_angles = Rmatrix.eulerAngles(2, 1, 0);
        euler_angles = NormalizeEuler_angles(euler_angles);
        cv::Mat middleVariables(1, 3, CV_32F, cv::Scalar(0));
        for (int k = 0; k < 3; k++) {
            middleVariables.at<float>(0, k) = euler_angles(k, 0);
            cout << "旋转轴角变量值为: " << middleVariables.at<float>(0, k) << endl;
        }
        cout << "t" << t << endl;
        RList.push_back(middleVariables);
        TList.push_back(t);
        GoodGrid.push_back(iGrid);
    } else {
        cout << "不能成功解算R,t" << endl;
    }
}


    for(int i=0;i<RList.size();i++)
    {
        float diff=MOdelDiffBetweenGrid(RList[1],RList[i],TList[1],TList[i]);
        cout<<"current grid : "<<GoodGrid[i]<<endl;
        cout<<"diff : "<<diff<<endl;
    }

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
