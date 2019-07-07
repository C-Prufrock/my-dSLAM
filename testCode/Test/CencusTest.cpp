//
// Created by true on 18-12-19.
//
#include "Tracking.h"  //Tracking.h中包含了Map.h Frame.h等等头文件,放心用数据.
#include "Converter.h"
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
#include <iomanip>


using namespace std;
using namespace ORB_SLAM2;

int main() {
    //获取相机参数;

    string strSettingPath = "/home/fzj/VslamBasedStaticDescribeSet/MonoBSS/Monocular/TUM3.yaml";
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_32F);
    ORBExtractorPara *ORBExtractorParameter = new ORBExtractorPara;
    ReadFromYaml(strSettingPath, K, DistCoef, ORBExtractorParameter);
    cout << "K : " << K << endl;

    //建立循环读取图像的函数
    //首先获取图像的id,文件名;
    //用来读取图像序列.
    /*
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = "/home/fzj/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();
    cout<<"nImages: "<<nImages<<endl;

    //for(int ni =0;ni<nImages-4;ni++) {
    // 接收图像,提取特征96652
    int ni = 0;
    cout << "*********************************************************************************" << endl;
    cout << "current ni" << ni << endl;*/

        cv::Mat ImgRgb1 = cv::imread("/home/fzj/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb/1341846647.766393.png", CV_LOAD_IMAGE_UNCHANGED);
        //cout << "vstrImageFilenames[ni]" <<"  "<< vstrImageFilenames[17] << endl;
        cv::Mat ImgRgb2 = cv::imread("/home/fzj/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb/1341846647.934224.png", CV_LOAD_IMAGE_UNCHANGED);

        cv::Mat mImGray1;
        cv::Mat mImGray2;

        //图像转化为灰度图;
        cv::cvtColor(ImgRgb1, mImGray1, CV_RGB2GRAY);
        cv::cvtColor(ImgRgb2, mImGray2, CV_RGB2GRAY);
        cout << "Size of ImageRgb1: " << ImgRgb1.size() << endl;

        //建立特征提取器
        ORB_SLAM2::ORBextractor *mpORBextractor = new ORBextractor( 4 * ORBExtractorParameter->nFeatures,
                                                                   ORBExtractorParameter->fScaleFactor,
                                                                   ORBExtractorParameter->nLevels,
                                                                   ORBExtractorParameter->fIniThFAST,
                                                                   ORBExtractorParameter->fMinThFAST);

        //构造Frame类,主要是真的方便呀--  唯一在我们实验中不需要构造的是三个数据.即ORBVocabulary,bf,thDepth.后者均可以设0,词典提取.
        ORB_SLAM2::ORBVocabulary *mpORBVocabulary = new ORBVocabulary();
        float bf = 0.0;
        float thDepth = 0.0;
        ORB_SLAM2::Frame mCurrentFrame1 = ORB_SLAM2::Frame(mImGray1, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef,
                                                           bf, thDepth);
        ORB_SLAM2::Frame mCurrentFrame2 = ORB_SLAM2::Frame(mImGray2, 0.0, mpORBextractor, mpORBVocabulary, K, DistCoef,
                                                           bf, thDepth);
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

        int nmatches = matcher.SearchForInitialization(mCurrentFrame1, mCurrentFrame2, mvbPrevMatched, mvIniMatches,
                                                       100);
        cout << "nmatches" << "   " << nmatches << endl;

        ////画出格子;
        //每个bin的大小为bsize;//格子的大小 事实上,划分的3方式都将会对图像的解算有巨大的影响.
        int bsizex = 160;//160;
        int bsizey = 120;//120;
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
        vector<Match> mvMatches12;       //mvMatch存储了 在第一帧与第二帧匹配特征点各自的index.

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
                keypoints2.push_back(mCurrentFrame2.mvKeysUn[mvIniMatches[i]]); //存储了所有匹配特征点的二维坐标;
                mvMatches12.push_back(currentMatchindex);//mvMatches12存储了匹配上特征点在所有特征点中的index;
            } else
                continue;
        }
        //cout<<keypoints1.size()<<endl;
        cv::Mat FeatureImg;
        cv::drawKeypoints(ImgRgb1, keypoints1, FeatureImg, cv::Scalar(0, 255, 255), 0); //画格子中的特征点;
        cv::imshow("FeatureIMg :", FeatureImg);
        cv::imwrite("/home/fzj/VslamBasedStaticDescribeSet/Result/TUM/FeaturesInGrid5DF.jpg", FeatureImg);


        //用于显示两张匹配的图像;
        int width = ImgRgb1.cols;
        int height = ImgRgb1.rows;
        cv::Size wholeSize(width * 2, height);
        cv::Mat outImg(wholeSize, ImgRgb1.type());;
        DrawMatches(keypoints1, keypoints2, ImgRgb1, ImgRgb2, outImg);
        //cv::imshow("MatchesImg :", outImg);
        //cv::imwrite("MatchesImg.jpg", outImg);

        //初始化mGrid;
        //将匹配的特征点分配到Grid[i][j];
        vector<int> mGrid[nGridcols][nGridrows];

        for (int i = 0; i < mvIniMatches.size(); i++)  //利用匹配较好的点.
        {
            if (mvIniMatches[i] >= 0) {
                int posX = mCurrentFrame1.mvKeysUn[i].pt.x / bsizex;
                int posY = mCurrentFrame1.mvKeysUn[i].pt.y / bsizey;
                if (posX < 0 || posX >= nGridcols || posY < 0 || posY >= nGridrows) {
                    continue;
                } else {
                    mGrid[posX][posY].push_back(i);  //mGrid保存了第一帧中可找到匹配点的特征点的index.
                }
            }
        }
        cout<<"here is all right "<<endl;

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
        if (GridHas.size() < 4) {
            //return false;
            //continue;
        }
        cout << "包含特征点数大于8的格子数目: " << GridHas.size() << endl;

        /// 取一个格子的8个点 计算模型,首先验证自身模型是否正确(原因是svd分解,匹配误差以及离散采样导致模型解算误差),
        /// 多遍历几次,每一次遍历计算外点得分数,得分数过高,说明该格子的模型混乱度过高,故舍去.
        /// 为得到该阈值,我们计算格子中的采样模型,记录每个模型的外点分数之和.
        /// 如正确并在全局范围内,检验模型的适应度.
        /// 首先我们可以观察一下情况 - - -实际把主要的代码都码了


        //保留得到的F模型 R,T以及模型的内点;
        vector<cv::Mat> F21list;
        vector<vector<cv::Point2f>> vbMatchInliersList1;
        vector<vector<cv::Point2f>> vbMatchInliersList2;
        //vector<int> GoodGrid;

        //主循环,遍历每一个格子;
        for (int iGrid = 0; iGrid < GridHas.size(); iGrid++) {

            vector<cv::Point2f> vbMatchInliers1;
            vector<cv::Point2f> vbMatchInliers2;
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
            cv::Mat T2t = T2.t();
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
            int nMaxIterations = 2000;
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
            cv::Mat F21i = cv::Mat::zeros(3, 3, CV_32F);
            cv::Mat F21 = cv::Mat::zeros(3, 3, CV_32F);                 //F21i是随机取的点所解算的模型,F21是获得的得分最高的模型;

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
                vector<bool> vbCurrentInliersInAllFeatures(mCurrentFrame1.mvKeysUn.size(),
                                                           false);   //当前ransac解算运动模型包含的内点,以格子中数目为其大小;
                //获取8个点.
                for (int j = 0; j < 8; j++) {
                    int idx = mvSets[it][j];
                    //cout<<"idx : "<<idx<<endl;
                    vPn1i[j] = vPn1[idx];
                    vPn2i[j] = vPn2[idx];
                    //cout<<"vPn1i[j]"<<vPn1i[j]<<endl;
                    //cout<<"vPn2i[j]"<<vPn2i[j]<<endl;
                }

                cv::Mat Fn = ComputeF21(vPn1i, vPn2i);
                F21i = T2t * Fn * T1;
                //cout<<"H21: "<<"  "<<H21i<<endl;

                //minSetScore.resize(8);
                //计算8个最小集合与模型的拟合度如何,如果过差则舍弃.

                //currentScore= CheckFundamentalBasedOnDistribution(F21i,F21,nmatches,keypoints1,keypoints2,3.87,sigma,vbCurrentInliers,KeypointInliers,it);
                //currentScore=CheckFundamentalBasedOnModel(F21i,F21,nmatches,keypoints1,keypoints2,3.84,sigma,vbCurrentInliers,KeypointInliers,it);
                currentScore = CheckDominatFundamental(F21i, mvIniMatches, mCurrentFrame1.mvKeysUn, GridHas[iGrid],
                                                       mCurrentFrame2.mvKeysUn, 2, sigma, vbCurrentInliers,
                                                       vbCurrentInliersInAllFeatures, it);
                //currentScore = CheckDominatHomograph(H21i, H12i, mvIniMatches, mCurrentFrame1.mvKeysUn, GridHas[iGrid],mCurrentFrame2.mvKeysUn, 3.871, sigma, vbCurrentInliers,vbCurrentInliersInAllFeatures,it);
                //cout << currentScore << endl;
                /*
                 * 保留最大分数,进入下一步循环.
                 */
                if (currentScore > score) {
                    F21 = F21i;
                    //cout<<"H21 : "<<"  "<<endl<<H21<<endl;
                    vbMatchesInliers = vbCurrentInliers;
                    vbMatchesInliersInAllFeatures = vbCurrentInliersInAllFeatures;
                    score = currentScore;
                    bestidx = it;
                    //MaxminSetScore=minSetScore;
                }
            }


            /// 画出解算的模型内点,或者说静态点,表为红色. 在ImgRgb1 上进行画图,该图已经画了格子;

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
            } //不必要再利用所有内点进行计算,最后利用所有内点进行优化即可.
            if (inliersnumber > 8) {

                cv::Mat Fm = ComputeF21(vPnInliers1, vPnInliers2);
                cout << "Grid index : " << "  " << iGrid << endl;
                cout << "inliersnumber:  " << inliersnumber << endl;
                //cout << "格子中解算得到的F模型为: " << endl << F21Final << endl;
                cv::Mat F21Final = T2t * Fm * T1;

                F21list.push_back(F21Final);
            } else {
                cout << " 模型内点数目为" << inliersnumber << " 求解F模型不正确 " << endl;
                continue;
            }

            //存储每个格子中的模型特征点；
            int inliersnumber2 = 0;

            for (int i = 0; i < vbMatchesInliersInAllFeatures.size(); i++) {
                if (vbMatchesInliersInAllFeatures[i]) {
                    //cout<<" inlier index in all features :"<< i <<endl;
                    cv::Point2f p1, p2;
                    p1.x = mCurrentFrame1.mvKeysUn[i].pt.x;
                    p1.y = mCurrentFrame1.mvKeysUn[i].pt.y;
                    p2.x = mCurrentFrame2.mvKeysUn[mvIniMatches[i]].pt.x;
                    p2.y = mCurrentFrame2.mvKeysUn[mvIniMatches[i]].pt.y;
                    vbMatchInliers1.push_back(p1);
                    vbMatchInliers2.push_back(p2);
                    circle(ImgRgb1, p1, 4, cv::Scalar(0, 255, 255));
                    inliersnumber2++;
                }
            }
            vbMatchInliersList1.push_back(vbMatchInliers1);
            vbMatchInliersList2.push_back(vbMatchInliers2);
            //cout<<" vbMatchesInliersInAllFeatures.size()"<<" "<< vbMatchesInliersInAllFeatures.size()<<endl<<"inliersnumber2 :"<<"  "<<inliersnumber2<<endl;
            cv::imshow("ImgRgb1 inlers:", ImgRgb1);
            cv::imwrite("/home/fzj/VslamBasedStaticDescribeSet/Result/TUM/ImageInliers5DF.jpg",ImgRgb1);


        }


        cout << " 共得到了" << " " << F21list.size() << " 个模型 " << endl;
        //cout<<"H21list [0].size() : "<<endl<<H21list[0].size()<<endl;
        //cout<<"vbMatchInliersList1[0] :"<< vbMatchInliersList1[0]<<endl;
        //cout<<GridHas[iGrid][0]<<endl;

        /*
        for(int i=0;i<GridHas[iGrid].size();i++)
        {
            cout<<"实际坐标为:   "<<mCurrentFrame1.mvKeysUn[GridHas[iGrid][i]].pt.x<<endl;
            cout<<"实际坐标为:   "<<mCurrentFrame1.mvKeysUn[GridHas[iGrid][i]].pt.y<<endl;
        }*/

        //cout<<"vbMatchInliersList2[0] :"<< vbMatchInliersList2[0].size()<<endl;
        //cout<<"H21list[0] : "<<H21list[0]<<endl;
        //验证模型内点之间的相互符合程度;
        //记录格子中的模型内点;
        cout << "F21list.size() : " << F21list.size() << endl;
        cv::Mat coupleMatrix = cv::Mat::zeros(F21list.size(), F21list.size(), CV_32F);
        //cout<<"H21listinv : "<<H21listinv<<endl;
        for (int i = 0; i < F21list.size(); i++) {
            for (int j = 0; j < F21list.size(); j++) {
                //cout<<"check number :"<<checkModelCoupling(H21list[i],H21listinv,vbMatchInliersList1[j],vbMatchInliersList2[j])<<endl;
                coupleMatrix.at<float>(i, j) = checkModelCouplingF(F21list[i], vbMatchInliersList1[j],
                                                                   vbMatchInliersList2[j]);
                //coupleMatrix.at<float>(i,j)=coupleMatrix.at<float>(i,j)/vbMatchInliersList1[i].size();
            }
        }
        for (int i = 0; i < vbMatchInliersList1.size(); i++) {
            cout << "vbMatchInliersList1[Good] :" << vbMatchInliersList1[i].size() << endl;
        }

        cout << "coupleMatrix : " << coupleMatrix << endl;
        //cout<<"coupleMatrix.cols"<<"   "<<coupleMatrix.cols<<endl;
        //遍历矩阵元素,获得每个模型在其他模型中的耦合度;
        cout << " coupleMatrix at [1,1]" << coupleMatrix.at<float>(1, 1) << "  " << endl;
        int j;
        for (int i = 0; i < coupleMatrix.cols; i++) {
            int maxCouple = vbMatchInliersList1[i].size();
            for (j = 0; j < coupleMatrix.rows; j++) {
                //cout<<" i :" <<i <<endl;
                //cout<<"j : "<<j <<endl;
                coupleMatrix.at<float>(j, i) = coupleMatrix.at<float>(j, i) / maxCouple;
            }
        }
        std::cout << std::fixed;   //固定小数点后数字的精度;
        cout << coupleMatrix << endl;

        //统计每行内欧和其他格子点数大于0.5的模型.或者视其他格子中的符合该模型的内点率大于0.5 即视为二格子模型耦合.
        //计算所有模型的耦合格子的分布.分布较为分散的视为静态特征集合的模型点.
        //那些与其他格子耦合度低且分布集中的模型视为outlier模型,不会采用该区域中的点进行模型估计.

        //首先计算每个格子特征点的质心,用来计算格子之间的分布程度.x
        //格子的质心 list;
        vector<cv::Point2f> meanOfGridFeaturelist;
        for (int i = 0; i < vbMatchInliersList1.size(); i++)  //遍历格子;
        {
            float sumx = 0;
            float sumy = 0;
            for (int j = 0; j < vbMatchInliersList1[i].size(); j++) {
                sumx += vbMatchInliersList1[i][j].x;
                sumy += vbMatchInliersList1[i][j].y;
            }
            sumx = sumx / vbMatchInliersList1[i].size();
            sumy = sumy / vbMatchInliersList1[i].size();
            cv::Point2f mean;
            mean.x = sumx;
            mean.y = sumy;
            meanOfGridFeaturelist.push_back(mean); //存储了每一个格子的质心;
        }
        /*
        for(int i=0;i<meanOfGridFeaturelist.size();i++)
        {
            cout<<"meanx of current grid"<<meanOfGridFeaturelist[i].x<<endl;
            cout<<"meanx of current grid"<<meanOfGridFeaturelist[i].y<<endl;
        }*/

        //计算每一个模型的耦合情况,如果在其他模型中找不到内点,找到的内点率小于0.5;
        // 说明该模型为outliers严重影响的模型或者动态模型,予以去除;
        //找到分布最广的格子模型;
        float thForSelectGrid = 0.65;
        vector<float> xvariancelist;  //用于保留每一个模型耦合的格子的分布情况,其分布用其质心的方差进行计算;
        vector<float> yvariancelist;
        for (int i = 0; i < coupleMatrix.rows; i++) {
            float meanx = 0;
            float meany = 0;
            float variancex = 0;
            float variancey = 0;
            int coupleGridnumber = 0;
            for (int j = 0; j < coupleMatrix.cols; j++) {
                if (coupleMatrix.at<float>(i, j) > thForSelectGrid) {  //如果耦合的格子符合该模型的点数大于0.5,则视为耦合格子,进行分布计算.
                    meanx += meanOfGridFeaturelist[j].x;
                    meany += meanOfGridFeaturelist[j].y;
                    coupleGridnumber++;
                }
            }
            cout << "coupleGridnumber : " << coupleGridnumber << endl;

            if (coupleGridnumber <= 1) {
                xvariancelist.push_back(0);
                yvariancelist.push_back(0);
                continue;
            }
            //如果有格子 则计算均值;
            meanx = meanx / coupleGridnumber;
            meany = meany / coupleGridnumber;

            //计算方差;差的平方和;
            for (int j = 0; j < coupleMatrix.cols; j++) {
                if (coupleMatrix.at<float>(i, j) > thForSelectGrid) {
                    variancex += pow((meanOfGridFeaturelist[j].x - meanx), 2);
                    variancey += pow((meanOfGridFeaturelist[j].y - meany), 2);
                }
            }
            variancex = sqrt(variancex);
            variancey = sqrt(variancey);

            cout << "variancex " << variancex << endl;
            cout << "variancey " << variancey << endl;
            xvariancelist.push_back(variancex);
            yvariancelist.push_back(variancey);
        }

        vector<float> valueVariance;
        int bestGridModel;
        float currentvariance = 0;
        //计算总的方差值,并比较出最大的方差.
        for (int i = 0; i < xvariancelist.size(); i++) {
            cout << "xvariance value: " << "  " << xvariancelist[i] << endl;
            float variance = sqrt((pow(xvariancelist[i], 2) + pow(yvariancelist[i], 2)));
            valueVariance.push_back(variance);
            if (variance > currentvariance) {
                currentvariance = variance;
                bestGridModel = i;
            }
        }
        cout << " best Grid model :" << bestGridModel << endl;

        //获取模型内点,并显示,注意.获取模型内点的规则;
        //取耦合的格子模型 在该模型内的内点也大于0.5的为静态特征集合.
        //以选取的格子的模型  选取耦合率大于0.5的格子,在这些格子中选取模型内点; 利用内点计算F模型;
        cout << "采用的格子的模型  : " << F21list[bestGridModel] << endl;

        //遍历那些耦合率大于0.5的格子,在其中寻找模型内点;

        vector<cv::Point2f> StaticSet1;
        vector<cv::Point2f> StaticSet2;
        float thForSeletFeature = 0.65;
        for (int i = 0; i < coupleMatrix.cols; i++) {
            vector<cv::Point2f> StatciSetCandidateInliers1;
            vector<cv::Point2f> StatciSetCandidateInliers2;
            if (coupleMatrix.at<float>(bestGridModel, i) > thForSeletFeature)  //如果矩阵耦合大于0.6;
            {
                int n = checkGoodPointInCoupleGrid(F21list[bestGridModel], vbMatchInliersList1[i],
                                                   vbMatchInliersList2[i], StatciSetCandidateInliers1,
                                                   StatciSetCandidateInliers2);
                cout << " number inliers: " << n << endl;
                for (int i = 0; i < StatciSetCandidateInliers1.size(); i++) {
                    cv::Point2f PointFortrans1;
                    cv::Point2f PointFortrans2;
                    PointFortrans1.x = StatciSetCandidateInliers1[i].x;
                    PointFortrans1.y = StatciSetCandidateInliers1[i].y;
                    PointFortrans2.x = StatciSetCandidateInliers2[i].x;
                    PointFortrans2.y = StatciSetCandidateInliers2[i].y;
                    StaticSet1.push_back(PointFortrans1);
                    StaticSet2.push_back(PointFortrans2);
                }
            }
        }
        cout << StaticSet1.size() << endl;
        //寻找模型内点;

        //cout<<"StatciSetCandidateInliers size : "<<StatciSetCandidateInliers1.size()<<endl;
        //checkGoodPointInCoupleGrid(F21list[bestGridModel],StatciSetCandidateInliers1,StatciSetCandidateInliers2,
        //StaticSet1,StaticSet2);
        //cout<<"StaticSet1.size() :  "<<StaticSet1.size()<<endl;
        //画出静态点,也可以进行10次local的优化后画静态点;
        for (int i = 0; i < StaticSet1.size(); i++) {

            //cout<<" inlier index in all features :"<< i <<endl;
            cv::Point2f p1, p2;
            p1.x = StaticSet1[i].x;
            p1.y = StaticSet1[i].y;
            p2.x = StaticSet2[i].x;
            p2.y = StaticSet2[i].y;
            circle(ImgRgb1, p1, 4, cv::Scalar(0, 255, 0));

        }
        cv::imshow("StaticPoint in Img1: ", ImgRgb1);
        cv::imwrite("/home/fzj/VslamBasedStaticDescribeSet/Result/TUM/CoupledMethod5DF.jpg", ImgRgb1);


        //最后进行非线性优化
        //计算模型内点;
        cv::waitKey(0);


    return 0;
}

