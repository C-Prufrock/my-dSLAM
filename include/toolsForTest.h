//
// Created by true on 18-12-11.
//

#ifndef VSLAMBASEDSTATICDESCRIBESET_TOOLSFORTEST_H
#define VSLAMBASEDSTATICDESCRIBESET_TOOLSFORTEST_H


#include<iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
void DrawMatches(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,cv::Mat ImgRgb1,cv::Mat ImgRgb2,cv::Mat& outImg);

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

void ReadFromYaml(string& strSettingPath,cv::Mat& K,cv::Mat& DistCoef,ORBExtractorPara* ORBExtractorParameter);

void DrawDescriptorsFLow(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,cv::Mat& outImg);

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

cv::Mat ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2);

cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

float CheckFundamentalBasedOnDistribution(cv::Mat F21i,cv::Mat F21,int nmatches,vector<cv::KeyPoint>keypoints1,
                       vector<cv::KeyPoint>keypoints2,float th,float sigma,vector<bool> &vbCurrentInliers,
                       vector<cv::KeyPoint>&KeypointInliers,int iterations);
float CheckFundamentalBasedOnModel(cv::Mat F21i,cv::Mat F21,int nmatches,vector<cv::KeyPoint>keypoints1,
                                         vector<cv::KeyPoint>keypoints2,float th,float sigma,vector<bool> &vbCurrentInliers,
                                         vector<cv::KeyPoint>&KeypointInliers,int iterations);
bool checkMaxminSets(cv::Mat F21i,vector<cv::Point2f> vPn1i,vector<cv::Point2f> vPn2i);

bool checkMaxminSetsForH(cv::Mat H21i,cv::Mat H12i,vector<cv::Point2f>vPn1i,vector<cv::Point2f>vPn2i);
float CheckDominatFundamental(cv::Mat F21i,vector<int> mvIniMatches,vector<cv::KeyPoint>keypoints1,vector<int>GridHasF,vector<cv::KeyPoint>keypoints2,float th,float sigma,vector<bool> &vbCurrentInliers, vector<bool>&vbCurrentInliersInAllFeatures,int iterations);
float CheckDominatHomograph(cv::Mat H21i,cv::Mat H12i,vector<int> mvIniMatches,vector<cv::KeyPoint>keypoints1,vector<int>GridHasF, vector<cv::KeyPoint>keypoints2,float th,float sigma,vector<bool> &vbCurrentInliers,vector<bool> &vbCurrentInliersInAllFeatures,int iterations);

void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

int CheckRT(cv::Mat &R, cv::Mat &t, vector<cv::KeyPoint> &vKeys1, vector<cv::KeyPoint> &vKeys2,cv::Mat &K);

void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

bool ComputeRTForF(cv::Mat &R,cv::Mat &t,cv::Mat F,vector<cv::KeyPoint>keypoints1,vector<cv::KeyPoint>keypoints2,
                   vector<bool> mvIniMatches,cv::Mat K);

void pose_estimation_2d2d (std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2, std::vector< cv::DMatch > matches,cv::Mat &F);

int checkModelCouplingH(cv::Mat H21i,cv::Mat H12i,vector<cv::Point2f>Points1,vector<cv::Point2f>Points2);

int checkModelCouplingF(cv::Mat F21i,vector<cv::Point2f>Points1,vector<cv::Point2f>Points2);

int checkGoodPointInCoupleGrid(cv::Mat F21i,vector<cv::Point2f>keypointInCoupleGrids1,vector<cv::Point2f>keypointInCoupleGrids2, vector<cv::Point2f>&StaticSet1,vector<cv::Point2f>&StaticSet2);

float CheckFundamental(const cv::Mat &F21,vector<cv::KeyPoint>&keypoints1,vector<cv::KeyPoint>&keypoints2,vector<bool>&vbCurrentInliers);

float DistributionCalculate(vector<cv::KeyPoint>keypoints1);
#endif //VSLAMBASEDSTATICDESCRIBESET_TOOLSFORTEST_H

