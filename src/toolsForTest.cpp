//
// Created by true on 18-12-11.
//

#include "toolsForTest.h"
#include<fstream>
#include<chrono>
#include<math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

void DrawMatches(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,cv::Mat ImgRgb1,cv::Mat ImgRgb2,cv::Mat& outImg)
{

    //构造outImg
    int width=ImgRgb1.cols;
    int height=ImgRgb1.rows;
    cv::Mat roi1=outImg(cv::Rect(0,0,width,height));
    cv::Mat roi2=outImg(cv::Rect(width,0,width,height));  //后面两个数是窗口的宽和高.
    ImgRgb1.copyTo(roi1);
    ImgRgb2.copyTo(roi2);

    int keypointsNumber = keypoints1.size();

    //画出特征点并连接二者
    for(int i=0;i<=keypointsNumber;i++)
    {
        cv::Point p1;
        p1.x=keypoints1[i].pt.x;
        p1.y=keypoints1[i].pt.y;

        cv::Point p2;
        p2.x=keypoints2[i].pt.x+width;
        p2.y=keypoints2[i].pt.y;


        cv::circle(outImg,p1,3,cv::Scalar(0, 255, 0));
        cv::circle(outImg,p2,3,cv::Scalar(0, 255, 0));
        cv::line(outImg,p1,p2,cv::Scalar(255,0,0));
    }
    //画出所有的特征点.
    //cv::Scalar matchColor=cv::Scalar::all(-1);
    //cv::Scalar singlePointColor=cv::Scalar::all(255);
    //cv::drawMatches(ImgRgb1,mCurrentFrame1.mvKeysUn,ImgRgb2,mCurrentFrame2.mvKeysUn,matches,outImg);
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
};

void DrawDescriptorsFLow(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,cv::Mat& outImg)
{
    //画出特征点并连接二者
    for(int i=0;i<=keypoints1.size();i++)
    {
        cv::Point2f p1;
        p1.x=keypoints1[i].pt.x;
        //cout<<"p1.x"<<p1.x<<endl;
        p1.y=keypoints1[i].pt.y;
        //cout<<"p1.y"<<p1.y<<endl;
        cv::Point2f p2;
        p2.x=keypoints2[i].pt.x;
        //cout<<"p2.x"<<p2.x<<endl;
        p2.y=keypoints2[i].pt.y;
        //cout<<"p2.y"<<p2.y<<endl;


        cv::circle(outImg,p1,3,cv::Scalar(0, 0, 255));
        cv::circle(outImg,p2,3,cv::Scalar(0, 255, 0));
        cv::line(outImg,p1,p2,cv::Scalar(255,0,0));
    }
};

void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    // 将所有vKeys点减去中心坐标，使x坐标和y坐标均值分别为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x); //返回浮点数的绝对值;
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // 将x坐标和y坐标分别进行尺度缩放，使得x坐标和y坐标的一阶绝对矩分别为1
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // |sX  0  -meanx*sX|
    // |0   sY -meany*sY|
    // |0   0      1    |
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
};

///计算F模型;
cv::Mat ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();
    //cout<<"N: "<<N<<endl;

    cv::Mat A(N,9,CV_32F); // N*9

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3); // v的最后一列 reshape函数将一行数据分为三行;获得F矩阵;

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0; // 秩2约束，将第3个奇异值设为0

    return  u*cv::Mat::diag(w)*vt;
};

///计算H模型;
cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F); // 2N*9

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i+1,0) = 0.0;
        A.at<float>(2*i+1,1) = 0.0;
        A.at<float>(2*i+1,2) = 0.0;
        A.at<float>(2*i+1,3) = u1;
        A.at<float>(2*i+1,4) = v1;
        A.at<float>(2*i+1,5) = 1;
        A.at<float>(2*i+1,6) = -v2*u1;
        A.at<float>(2*i+1,7) = -v2*v1;
        A.at<float>(2*i+1,8) = -v2;

        A.at<float>(2*i,0) = u1;
        A.at<float>(2*i,1) = v1;
        A.at<float>(2*i,2) = 1;
        A.at<float>(2*i,3) = 0.0;
        A.at<float>(2*i,4) = 0.0;
        A.at<float>(2*i,5) = 0.0;
        A.at<float>(2*i,6) = -u2*u1;
        A.at<float>(2*i,7) = -u2*v1;
        A.at<float>(2*i,8) = -u2;

    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3); // v的最后一列
};

bool checkMaxminSets(cv::Mat F21i,vector<cv::Point2f> vPn1i,vector<cv::Point2f> vPn2i){
    for(int j=0;j<8;j++) {
        const float th = 3.871;
        const float sigma =1.0;
        const float f11 = F21i.at<float>(0,0);
        const float f12 = F21i.at<float>(0,1);
        const float f13 = F21i.at<float>(0,2);
        const float f21 = F21i.at<float>(1,0);
        const float f22 = F21i.at<float>(1,1);
        const float f23 = F21i.at<float>(1,2);
        const float f31 = F21i.at<float>(2,0);
        const float f32 = F21i.at<float>(2,1);
        const float f33 = F21i.at<float>(2,2);
        const float u1 = vPn1i[j].x;
        const float v1 = vPn1i[j].y;
        const float u2 = vPn2i[j].x;
        const float v2 = vPn1i[j].y;
        //cout<<"u1 :"<<u1<<"v1:"<<v1<<"u2:"<<u2<<"v2"<<v2<<endl;

        const float invSigmaSquare = 1.0/(sigma*sigma);
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            return false;

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;
        if(chiSquare2>th)
            return false;
    }
    //cout<<"success sample"<<endl;
};

bool checkMaxminSetsForH(cv::Mat H21i,cv::Mat H12i,vector<cv::Point2f>vPn1i,vector<cv::Point2f>vPn2i){

    const float h11 = H21i.at<float>(0,0);
    const float h12 = H21i.at<float>(0,1);
    const float h13 = H21i.at<float>(0,2);
    const float h21 = H21i.at<float>(1,0);
    const float h22 = H21i.at<float>(1,1);
    const float h23 = H21i.at<float>(1,2);
    const float h31 = H21i.at<float>(2,0);
    const float h32 = H21i.at<float>(2,1);
    const float h33 = H21i.at<float>(2,2);

    // |h11inv h12inv h13inv|
    // |h21inv h22inv h23inv|
    // |h31inv h32inv h33inv|
    const float h11inv = H12i.at<float>(0,0);
    const float h12inv = H12i.at<float>(0,1);
    const float h13inv = H12i.at<float>(0,2);
    const float h21inv = H12i.at<float>(1,0);
    const float h22inv = H12i.at<float>(1,1);
    const float h23inv = H12i.at<float>(1,2);
    const float h31inv = H12i.at<float>(2,0);
    const float h32inv = H12i.at<float>(2,1);
    const float h33inv = H12i.at<float>(2,2);

    //获得图像坐标.
    const float th=5.991;
    const float sigma=1.0;
    const float invSigmaSquare = 1.0/(sigma*sigma);
    for(int j=0;j<8;j++) {

        const float u1 = vPn1i[j].x;
        const float v1 = vPn1i[j].y;
        const float u2 = vPn2i[j].x;
        const float v2 = vPn1i[j].y;

        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算重投影误差
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        // 根据方差归一化误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            return false;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // 将图像1中的特征点单应到图像2中
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;
        if(chiSquare2>th)
            return false;

    }
} ;


float CheckFundamentalBasedOnDistribution(cv::Mat F21i,cv::Mat F21,int nmatches,vector<cv::KeyPoint>keypoints1,
              vector<cv::KeyPoint>keypoints2,float th,float sigma,vector<bool> &vbCurrentInliers,
              vector<cv::KeyPoint>&KeypointInliers,int iterations)
{
    KeypointInliers.clear();
    const float f11 = F21i.at<float>(0,0);
    const float f12 = F21i.at<float>(0,1);
    const float f13 = F21i.at<float>(0,2);
    const float f21 = F21i.at<float>(1,0);
    const float f22 = F21i.at<float>(1,1);
    const float f23 = F21i.at<float>(1,2);
    const float f31 = F21i.at<float>(2,0);
    const float f32 = F21i.at<float>(2,1);
    const float f33 = F21i.at<float>(2,2);
    float currentScore =0;
    for(int i=0; i<nmatches; i++)
      {
        bool bIn=true;
        const cv::KeyPoint &kp1 = keypoints1[i];  //keypoints1 存储了参考帧中匹配的特征点
        const cv::KeyPoint &kp2 = keypoints2[i];  //keypoints2 存储了tracking 帧中对应的匹配点.

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;


        const float invSigmaSquare = 1.0/(sigma*sigma);
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
         // F21x1可以算出x1在图像中x2对应的线l
         const float a2 = f11*u1+f12*v1+f13;
         const float b2 = f21*u1+f22*v1+f23;
         const float c2 = f31*u1+f32*v1+f33;

         // x2应该在l这条线上:x2点乘l = 0
         const float num2 = a2*u2+b2*v2+c2;
         const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方
         const float chiSquare1 = squareDist1*invSigmaSquare;

         if(chiSquare1>th){
             bIn = false;
         }
         // Reprojection error in second image
         // l1 =x2tF21=(a1,b1,c1)
         const float a1 = f11*u2+f21*v2+f31;
         const float b1 = f12*u2+f22*v2+f32;
         const float c1 = f13*u2+f23*v2+f33;

         const float num1 = a1*u1+b1*v1+c1;

         const float squareDist2 = num1*num1/(a1*a1+b1*b1);

         const float chiSquare2 = squareDist2*invSigmaSquare;

         if(chiSquare2>th){
             bIn = false;
         }
         if(bIn){
             vbCurrentInliers[i]=true;
             KeypointInliers.push_back(kp1);
         }
         else{
             vbCurrentInliers[i]= false;
             continue;
         }
      }
      //cout<<"inliersnumber : "<<KeypointInliers.size()<<endl;
      /*
        * 利用keypointInliers计算当前采样模型的得分;
        *
        * */

      /*
       * * 第一步  获取x,y的均值;
       * * */
      float Xmean=0;
      float Ymean=0;
      float Xsum=0;
      float Ysum=0;
      float XDiffSqure=0;
      float YDiffSqure=0;

      for(int i=0;i<KeypointInliers.size();i++)
      {
          Xsum+=KeypointInliers[i].pt.x;
          Ysum+=KeypointInliers[i].pt.y;
      }
      Xmean=Xsum/KeypointInliers.size();
      Ymean=Ysum/KeypointInliers.size();

      /*
       * * 第二步,计算得分;
       * * */
      for(int i=0;i<KeypointInliers.size();i++)
      {
          XDiffSqure+=pow((KeypointInliers[i].pt.x-Xmean),2);
          YDiffSqure+=pow((KeypointInliers[i].pt.y-Ymean),2);
      }

      XDiffSqure=XDiffSqure/(KeypointInliers.size()-1);
      YDiffSqure=YDiffSqure/(KeypointInliers.size()-1);

      currentScore=sqrt(pow(XDiffSqure,2)+pow(YDiffSqure,2));

      //cout<<"current iterations"<<iterations<<endl;
      //cout<<"current score"<<currentScore<<endl;
      return currentScore;
};


///利用最小集合解算模型,看最小集合自身的拟合度,如果都在范围之内,说明离散误差,匹配误差较小,否则舍去该计算模型;
///bool checkFundamentalInsideMinSet();

float CheckFundamentalBasedOnModel(cv::Mat F21i,cv::Mat F21,int nmatches,vector<cv::KeyPoint>keypoints1,
        vector<cv::KeyPoint>keypoints2,float th,float sigma,vector<bool> &vbCurrentInliers,
        vector<cv::KeyPoint>&KeypointInliers,int iterations)
{
    float  const thScore = 5.991;    //最高得分,如果完全符合运动模型,则得5.991分.

    KeypointInliers.clear();
    const float f11 = F21i.at<float>(0,0);
    const float f12 = F21i.at<float>(0,1);
    const float f13 = F21i.at<float>(0,2);
    const float f21 = F21i.at<float>(1,0);
    const float f22 = F21i.at<float>(1,1);
    const float f23 = F21i.at<float>(1,2);
    const float f31 = F21i.at<float>(2,0);
    const float f32 = F21i.at<float>(2,1);
    const float f33 = F21i.at<float>(2,2);
    float currentScore =0;
    for(int i=0; i<nmatches; i++)
    {
        bool bIn=true;
        const cv::KeyPoint &kp1 = keypoints1[i];  //keypoints1 存储了参考帧中匹配的特征点
        const cv::KeyPoint &kp2 = keypoints2[i];  //keypoints2 存储了tracking 帧中对应的匹配点.

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        const float invSigmaSquare = 1.0/(sigma*sigma);
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方
        const float chiSquare1 = squareDist1*invSigmaSquare;


        if(chiSquare1>th){
            bIn = false;
        }else {
            currentScore+=thScore-chiSquare1;
        }

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;
        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;
        //cout<<"chiSquare1 :"<<"   "<<chiSquare1<<endl;
        //cout<<"chiSquare2:"<<chiSquare2<<endl;
        if(chiSquare2>th){
            bIn = false;
        }else{
            currentScore+=thScore-chiSquare2;
        }
        if(bIn){
            vbCurrentInliers[i]=true;
            KeypointInliers.push_back(kp1);
        }
        else{
            vbCurrentInliers[i]=false;
        }
    }
    //cout<<"inliersnumber : "<<"  "<<KeypointInliers.size()<<endl;
    //cout<<"current score"<<"  "<<currentScore<<endl;
    //cout<<"current Iterations: "<<"  "<<iterations<<endl;
    return currentScore;
}


float CheckDominatFundamental(cv::Mat F21i,vector<int> mvIniMatches,vector<cv::KeyPoint>keypoints1,vector<int>GridHasF, vector<cv::KeyPoint>keypoints2,float th,float sigma,vector<bool> &vbCurrentInliers,
                              vector<bool>&vbCurrentInliersInAllFeatures,int iterations)
{
    const float thScore=5.991;
    const float f11 = F21i.at<float>(0,0);
    const float f12 = F21i.at<float>(0,1);
    const float f13 = F21i.at<float>(0,2);
    const float f21 = F21i.at<float>(1,0);
    const float f22 = F21i.at<float>(1,1);
    const float f23 = F21i.at<float>(1,2);
    const float f31 = F21i.at<float>(2,0);
    const float f32 = F21i.at<float>(2,1);
    const float f33 = F21i.at<float>(2,2);
    float currentScore =0;
    //int inliersnumber=0;
    for(int i=0; i<GridHasF.size(); i++) //遍历格子中的特征点数目以计算得分;
    {
        bool bIn=true;
        const cv::KeyPoint &kp1 = keypoints1[GridHasF[i]];  //keypoints1 存储了参考帧中匹配的特征点
        //cout<<"keypoint1 position x: "<<keypoints1[GridHasF[i]].pt.x<<endl;
        //cout<<"keypoint position y: "<<keypoints1[GridHasF[i]].pt.y<<endl;
        const cv::KeyPoint &kp2 = keypoints2[mvIniMatches[GridHasF[i]]];  //keypoints2 存储了tracking 帧中对应的匹配点.
        //cout<<"keypoint2 position x: "<<keypoints2[GridHasF[i]].pt.x<<endl;
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        const float invSigmaSquare = 1.0/(sigma*sigma);
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方
        const float chiSquare1 = squareDist1*invSigmaSquare;
        //cout<<"chiSquare1 :"<<"   "<<chiSquare1<<endl;
        if(chiSquare1>th){
            bIn = false;
        }else {
            currentScore+=thScore-chiSquare1;
        }

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;


        //cout<<"chiSquare2:"<<chiSquare2<<endl;
        if(chiSquare2>th){
            bIn = false;
        }else{
            currentScore+=thScore-chiSquare2;
        }

        if(bIn){
            vbCurrentInliers[i]=true;
            //cout<<"GridHasF[i]"<<GridHasF[i]<<endl;
            vbCurrentInliersInAllFeatures[GridHasF[i]]=true;
        }
        else{
            vbCurrentInliers[i]=false;
            vbCurrentInliersInAllFeatures[GridHasF[i]]=false;
        }
    }
    //cout<<"inliersnumber : "<<"  "<<KeypointInliers.size()<<endl;
    //cout<<"current score"<<"  "<<currentScore<<endl;
    //cout<<"current Iterations: "<<"  "<<iterations<<endl;
    return currentScore;
};

float CheckDominatHomograph(cv::Mat H21i,cv::Mat H12i,vector<int> mvIniMatches,vector<cv::KeyPoint>keypoints1,vector<int>GridHasF, vector<cv::KeyPoint>keypoints2,float th,float sigma,vector<bool> &vbCurrentInliers,
                            vector<bool> &vbCurrentInliersInAllFeatures,int iterations){
    const float thScore=5.991;
    //cout<<"H21i : "<<endl<< H21i <<endl;
    //cout<<"H12i : "<<endl<< H12i <<endl;
    //KeypointInliers.clear();
    const float h11 = H21i.at<float>(0,0);
    const float h12 = H21i.at<float>(0,1);
    const float h13 = H21i.at<float>(0,2);
    const float h21 = H21i.at<float>(1,0);
    const float h22 = H21i.at<float>(1,1);
    const float h23 = H21i.at<float>(1,2);
    const float h31 = H21i.at<float>(2,0);
    const float h32 = H21i.at<float>(2,1);
    const float h33 = H21i.at<float>(2,2);

    // |h11inv h12inv h13inv|
    // |h21inv h22inv h23inv|
    // |h31inv h32inv h33inv|
    const float h11inv = H12i.at<float>(0,0);
    const float h12inv = H12i.at<float>(0,1);
    const float h13inv = H12i.at<float>(0,2);
    const float h21inv = H12i.at<float>(1,0);
    const float h22inv = H12i.at<float>(1,1);
    const float h23inv = H12i.at<float>(1,2);
    const float h31inv = H12i.at<float>(2,0);
    const float h32inv = H12i.at<float>(2,1);
    const float h33inv = H12i.at<float>(2,2);
    float currentScoreH =0;
    //int inliersnumber=0;
    for(int i=0; i<GridHasF.size(); i++)
    {
        bool bIn=true;
        const cv::KeyPoint &kp1 = keypoints1[GridHasF[i]];  //keypoints1 存储了参考帧中匹配的特征点
        //cout<<"keypoint position x: "<<keypoints1[GridHasF[i]].pt.x<<endl;
        //cout<<"keypoint position y: "<<keypoints1[GridHasF[i]].pt.y<<endl;
        const cv::KeyPoint &kp2 = keypoints2[mvIniMatches[GridHasF[i]]];  //keypoints2 存储了tracking 帧中对应的匹配点.

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;

        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        //cout<<"u1 :"<<u1<<endl<<"v1 :"<<v1<<endl;
        //cout<<"u2 :"<<u2<<endl<<"v2 :"<<v2<<endl;

        const float invSigmaSquare = 1.0/(sigma*sigma);


        // Reprojection error in first image
        // x2in1 = H12*x2
        // 将图像2中的特征点单应到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|
        // |v1| = |h21inv h22inv h23inv||v2|
        // |1 |   |h31inv h32inv h33inv||1 |
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        //cout<<"w2in1inv : "<<w2in1inv<<endl;
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算重投影误差
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        //cout<<"squareDist1 :"<<"  "<<squareDist1<<endl;
        // 根据方差归一化误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>thScore)
            bIn = false;
        else
            currentScoreH += thScore - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // 将图像1中的特征点单应到图像2中
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        //cout<<"squareDist2:"<<"  "<<squareDist2<<endl;
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>thScore) //所谓卡方检验 意味这该数据有多少概率符合该模型;
            bIn = false;
        else
            currentScoreH += thScore - chiSquare2;
        if(bIn){
            vbCurrentInliers[i]=true;  //返回的是所有特征点的坐标.
            vbCurrentInliersInAllFeatures[GridHasF[i]]=true;
            //KeypointInliers.push_back(kp1);      //获得所有为内点的特征点;
        }
        else{
            vbCurrentInliers[i]=false;
            vbCurrentInliersInAllFeatures[GridHasF[i]]=false;
        }
    }
    //cout<<"inliersnumber : "<<"  "<<KeypointInliers.size()<<endl;
    //cout<<"current score"<<"  "<<currentScore<<endl;
    //cout<<"current Iterations: "<<"  "<<iterations<<endl;
    return currentScoreH;
}
void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    // 对 t 有归一化，但是这个地方并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0) // 旋转矩阵有行列式为1的约束
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
};

int CheckRT(cv::Mat &R, cv::Mat &t, vector<cv::KeyPoint> &vKeys1, vector<cv::KeyPoint> &vKeys2,cv::Mat &K)
{

    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    const float th2=3.874;
    // Camera 1 Projection Matrix K[I|0]
    // 步骤1：得到一个相机的投影矩阵
    // 以第一个相机的光心作为世界坐标系
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    // 第一个相机的光心在世界坐标系下的坐标
    //cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    // 步骤2：得到第二个相机的投影矩阵
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;
    // 第二个相机的光心在世界坐标系下的坐标
    //cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vKeys1.size();i<iend;i++)
    {
        // kp1和kp2是匹配特征点
        const cv::KeyPoint &kp1 = vKeys1[i];
        const cv::KeyPoint &kp2 = vKeys2[i];
        cv::Mat p3dC1;

        // 步骤3：利用三角法恢复三维点p3dC1
        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            continue;
        }
        // 步骤5：判断3D点是否在两个摄像头前方

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 步骤5.1：3D点深度为负，在第一个摄像头后方，淘汰
        if(p3dC1.at<float>(2)<=0 )
            continue;
        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 步骤5.2：3D点深度为负，在第二个摄像头后方，淘汰
        cv::Mat p3dC2 = R*p3dC1+t;

        //计算重投影误差,如果误差过大,则淘汰;
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
        //cout<<"squareError1  :  "<<squareError1<<endl;

        if(squareError1>th2)
        {
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
            continue;
        }
        if(p3dC2.at<float>(2)<=0 )
            continue;

        nGood++;
    }

    return nGood;
};

//本函数本来用于格子中的运动模型解算;因此本来还有描述格子特征点的参数；
//目前仍是检验F矩阵分解出来的R,T,因此不再需要该参数了.

bool ComputeRTForF(cv::Mat &R,cv::Mat &t,cv::Mat F,vector<cv::KeyPoint>keypoints1,vector<cv::KeyPoint>keypoints2,
                vector<bool> mvIniMatches,cv::Mat K)
{
    cv::Mat E21=K.t()*F*K;
    cv::Mat R1,R2,tn;
    DecomposeE(E21,R1,R2,tn);
    cv::Mat t1=tn;
    cv::Mat t2=-tn;

    vector<cv::KeyPoint>currentGridFeatures1;
    currentGridFeatures1.clear();
    vector<cv::KeyPoint>currentGridFeatures2;
    currentGridFeatures2.clear();
    //cout<<"vbMatchesInliersForModel.size(): "<<" "<<vbMatchesInliersForModel.size()<<endl;
    for(int i=0;i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i])
        {
            currentGridFeatures1.push_back(keypoints1[i]);
            //cout<<"inliers index in img1 :"<<" "<<i<<endl;
            currentGridFeatures2.push_back(keypoints2[i]);
        }
    }
    cout<<"用于检验Ｒ，Ｔ的特征点个数：   "<<currentGridFeatures1.size()<<endl;

    int nGood1 = CheckRT(R1,t1,currentGridFeatures1,currentGridFeatures2,K);
    int nGood2 = CheckRT(R2,t1,currentGridFeatures1,currentGridFeatures2,K);
    int nGood3 = CheckRT(R1,t2,currentGridFeatures1,currentGridFeatures2,K);
    int nGood4 = CheckRT(R2,t2,currentGridFeatures1,currentGridFeatures2,K);

    //cout<<"current index of F21"<<"  "<<nF<<endl;
    cout<<"mGood1"<<"  "<<nGood1<<endl;
    cout<<"mGood2"<<"  "<<nGood2<<endl;
    cout<<"mGood3"<<"  "<<nGood3<<endl;
    cout<<"mGood4"<<"  "<<nGood4<<endl;
    //cout<<"size of R" <<R1.size()<<endl;
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));  //取其中能被较好三角化的点.

    if(maxGood<8)
        return false;

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    if(nsimilar>1)
    {
        cout<<"两组R,t符合要求,因此求解失败;"<<endl;
        return false;
    }

    if(maxGood==nGood1){
        R=R1;
        t=t1;
    }else if(maxGood==nGood2)
    {
        R=R2;
        t=t1;
    }else if(maxGood==nGood3)
    {
        R=R1;
        t=t2;
    }else if(maxGood==nGood4)
    {
        R=R2;
        t=t2;
    }
    return true;
};


void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    // 在DecomposeE函数和ReconstructH函数中对t有归一化
    // 这里三角化过程中恢复的3D点深度取决于 t 的尺度，
    // 但是这里恢复的3D点并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变

    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
};


void pose_estimation_2d2d ( std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2, std::vector< DMatch > matches,Mat& F)
{
    // 相机内参,TUM Freiburg2
    Mat K = ( Mat_<double> ( 3,3 ) << 535.4, 0,320.1, 0, 539.2, 247.6, 0, 0, 1 );

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵
    F= findFundamentalMat ( points1, points2, CV_FM_8POINT );
    cout<<"fundamental_matrix is "<<endl<< F<<endl;

};

int checkModelCouplingH(cv::Mat H21i,cv::Mat H12i,vector<cv::Point2f>Points1,vector<cv::Point2f>Points2)
{
    int couplingnumber=0;
    //cout<<"H21i : "<<endl<< H21i <<endl;
    //cout<<"H12i : "<<endl<< H12i <<endl;
    float thScore=5.9991;
    const float h11 = H21i.at<float>(0,0);
    const float h12 = H21i.at<float>(0,1);
    const float h13 = H21i.at<float>(0,2);
    const float h21 = H21i.at<float>(1,0);
    const float h22 = H21i.at<float>(1,1);
    const float h23 = H21i.at<float>(1,2);
    const float h31 = H21i.at<float>(2,0);
    const float h32 = H21i.at<float>(2,1);
    const float h33 = H21i.at<float>(2,2);

    // |h11inv h12inv h13inv|
    // |h21inv h22inv h23inv|
    // |h31inv h32inv h33inv|
    const float h11inv = H12i.at<float>(0,0);
    const float h12inv = H12i.at<float>(0,1);
    const float h13inv = H12i.at<float>(0,2);
    const float h21inv = H12i.at<float>(1,0);
    const float h22inv = H12i.at<float>(1,1);
    const float h23inv = H12i.at<float>(1,2);
    const float h31inv = H12i.at<float>(2,0);
    const float h32inv = H12i.at<float>(2,1);
    const float h33inv = H12i.at<float>(2,2);

    for(int i=0; i<Points1.size(); i++)
    {
        bool bIn=true;
        const cv::Point2f &kp1 = Points1[i];  //keypoints1 存储了参考帧中匹配的特征点
        //cout<<"keypoint position x: "<<keypoints1[GridHasF[i]].pt.x<<endl;
        //cout<<"keypoint position y: "<<keypoints1[GridHasF[i]].pt.y<<endl;
        const cv::Point2f &kp2 = Points2[i];  //keypoints2 存储了tracking 帧中对应的匹配点.

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;
        //cout<<"u1 :"<<u1 <<endl<<"v1 :"<<v1<<endl;
        //cout<<"u2 :"<<u2 <<endl<<"v2 :"<<v2<<endl;
        const float invSigmaSquare = 1.0;


        // Reprojection error in first image
        // x2in1 = H12*x2
        // 将图像2中的特征点单应到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|
        // |v1| = |h21inv h22inv h23inv||v2|
        // |1 |   |h31inv h32inv h33inv||1 |
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        //cout<<"w2in1inv : "<<w2in1inv<<endl;
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算重投影误差
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        //cout<<"squareDist1 :"<<"  "<<squareDist1<<endl;
        // 根据方差归一化误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>thScore)
            bIn = false;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // 将图像1中的特征点单应到图像2中
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        //cout<<"squareDist2:"<<"  "<<squareDist2<<endl;
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>thScore) //所谓卡方检验 意味这该数据有多少概率符合该模型;
            bIn = false;
        if(bIn){
            couplingnumber++;
        }
    }
    return couplingnumber;
};

int checkModelCouplingF(cv::Mat F21i,vector<cv::Point2f>Points1,vector<cv::Point2f>Points2)
{
    int couplingnumber=0;
    const float thScore=5.991;
    const float f11 = F21i.at<float>(0,0);
    const float f12 = F21i.at<float>(0,1);
    const float f13 = F21i.at<float>(0,2);
    const float f21 = F21i.at<float>(1,0);
    const float f22 = F21i.at<float>(1,1);
    const float f23 = F21i.at<float>(1,2);
    const float f31 = F21i.at<float>(2,0);
    const float f32 = F21i.at<float>(2,1);
    const float f33 = F21i.at<float>(2,2);
    float currentScore =0;
    //int inliersnumber=0;
    for(int i=0; i<Points1.size(); i++) //遍历格子中的特征点数目以计算得分;
    {
        bool bIn=true;
        const cv::Point2f &kp1 = Points1[i];  //keypoints1 存储了参考帧中匹配的特征点
        //cout<<"keypoint position x: "<<keypoints1[GridHasF[i]].pt.x<<endl;
        //cout<<"keypoint position y: "<<keypoints1[GridHasF[i]].pt.y<<endl;
        const cv::Point2f &kp2 = Points2[i];  //keypoints2 存储了tracking 帧中对应的匹配点.

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

        const float invSigmaSquare = 1.0;
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方
        const float chiSquare1 = squareDist1*invSigmaSquare;
        //cout<<"chiSquare1 :"<<"   "<<chiSquare1<<endl;
        if(chiSquare1>thScore){
            bIn = false;
        }
        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;


        //cout<<"chiSquare2:"<<chiSquare2<<endl;
        if(chiSquare2>thScore){
            bIn = false;
        }

        if(bIn){

           couplingnumber++;
        }
    }
    return couplingnumber;
};



int checkGoodPointInCoupleGrid(cv::Mat F21i,vector<cv::Point2f>keypointInCoupleGrids1,vector<cv::Point2f>keypointInCoupleGrids2, vector<cv::Point2f>&StaticSet1,vector<cv::Point2f>&StaticSet2)
{
    const float thScore=5.991;
    const float f11 = F21i.at<float>(0,0);
    const float f12 = F21i.at<float>(0,1);
    const float f13 = F21i.at<float>(0,2);
    const float f21 = F21i.at<float>(1,0);
    const float f22 = F21i.at<float>(1,1);
    const float f23 = F21i.at<float>(1,2);
    const float f31 = F21i.at<float>(2,0);
    const float f32 = F21i.at<float>(2,1);
    const float f33 = F21i.at<float>(2,2);
    int StaticSet=0;
    for(int i=0; i<keypointInCoupleGrids1.size(); i++) //遍历格子中的特征点数目以计算得分;
    {

        bool bIn=true;
        const cv::Point2f &kp1 = keypointInCoupleGrids1[i];  //keypoints1 存储了参考帧中匹配的特征点
        //cout<<"keypoint position x: "<<keypoints1[GridHasF[i]].pt.x<<endl;
        //cout<<"keypoint position y: "<<keypoints1[GridHasF[i]].pt.y<<endl;
        const cv::Point2f &kp2 = keypointInCoupleGrids2[i];  //keypoints2 存储了tracking 帧中对应的匹配点.

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

        const float invSigmaSquare = 1.0;
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方
        const float chiSquare1 = squareDist1*invSigmaSquare;
        //cout<<"chiSquare1 :"<<"   "<<chiSquare1<<endl;
        if(chiSquare1>thScore){
            bIn = false;
        }
        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;


        //cout<<"chiSquare2:"<<chiSquare2<<endl;
        if(chiSquare2>thScore){
            bIn = false;
        }

        if(bIn){
            StaticSet++;
            StaticSet1.push_back(kp1);
            StaticSet2.push_back(kp2);
        }
    }
    return StaticSet;
};

float CheckFundamental(const cv::Mat &F21i,vector<cv::KeyPoint>&keypoints1,vector<cv::KeyPoint>&keypoints2,vector<bool>&vbCurrentInliers)
{
    const int N = keypoints1.size();

    const float f11 = F21i.at<float>(0,0);
    const float f12 = F21i.at<float>(0,1);
    const float f13 = F21i.at<float>(0,2);
    const float f21 = F21i.at<float>(1,0);
    const float f22 = F21i.at<float>(1,1);
    const float f23 = F21i.at<float>(1,2);
    const float f31 = F21i.at<float>(2,0);
    const float f32 = F21i.at<float>(2,1);
    const float f33 = F21i.at<float>(2,2);

    float score = 0.0;
    const float th = 3.871;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0;

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = keypoints1[i];
        const cv::KeyPoint &kp2 = keypoints2[i];;
        //cout<<"kp1 : "<<kp1.pt.x<<"   "<<kp1.pt.y<<endl;
        //cout<<"kp2 : "<<kp2.pt.x<<"   "<<kp2.pt.y<<endl;

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0
        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方

        const float chiSquare1 = squareDist1*invSigmaSquare;
        //cout<<"chiSquare1 : "<<chiSquare1<<endl;
        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;
        //cout<<"chiSquare2 : "<<  chiSquare2   <<endl;
        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbCurrentInliers[i]=true;
        else
            vbCurrentInliers[i]=false;
    }

    return score;
}

float DistributionCalculate(vector<cv::KeyPoint>keypoints1)
{
    //如果keypoints为空,此时返回分布度为0；
    if(keypoints1.size()==0)
        return 0.0;
    float Distribution=0.0;
    float meanx=0.0;
    float meany=0.0;
    for(int i=0;i<keypoints1.size();i++)
    {
        meanx+=keypoints1[i].pt.x;
        meany+=keypoints1[i].pt.y;
    }
    meanx=meanx/keypoints1.size();
    meany=meany/keypoints1.size();

    float variancex=0.0;
    float variancey=0.0;
    for(int i=0;i<keypoints1.size();i++)
    {
        variancex+=pow((keypoints1[i].pt.x-meanx),2);
        variancey+=pow((keypoints1[i].pt.y-meany),2);
    }
    Distribution=sqrt((variancex+variancey)/keypoints1.size());
    cout<<"Distribution  : "<<Distribution<<endl;

    return Distribution;

}
//计算分布后的得分;

