#include <iostream>
#include <opencv2/opencv.hpp>
#include "ORBextractor.h"
#include "ORBmatcher.h"


using namespace std;
using namespace cv;

//设置extractor基本参数,该过程为yaml文件中的基本参数;
int nfeatures=1000;
double scaleFactor=1.2;
int nlevels=8;
int iniThFAST=50;
int minThFAST=30;


int main() {
    std::cout << "Hello, World!" << std::endl;
    Mat img=imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/data/timg.jpeg");
    Mat img2=imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/data/1341846227.218443.png",1);

    waitKey(0);
    cout<<img.channels()<<endl;
    //cout<<img.size()<<endl;
    Mat gray_image;
    Mat gray_image2;
    cvtColor(img,gray_image,CV_BGR2GRAY);
    cvtColor(img2,gray_image2,CV_BGR2GRAY);
    //cout<<gray_image.size()<<endl;



    //进行orbextractor提取;
    std::vector<cv::KeyPoint>keypoints,keypoints2;
    Mat mask;
    Mat descriptors,descriptors2;
    ORB_SLAM2::ORBextractor orbextractor(nfeatures,scaleFactor,nlevels,iniThFAST,minThFAST);
    orbextractor(gray_image,mask,keypoints,descriptors);
    //orbextractor(gray_image2,mask,keypoints2,descriptors2);

    //计算每层应提取多少特征点
    for(int i=1;i<nlevels;i++)
    {cout<<orbextractor.GetmnFeaturePerLevel().at(i)<<endl;}

    //test图像金字塔;


    //计算匹配距离;
    ORB_SLAM2::ORBmatcher Match;
    //输出描述子矩阵的size;
    //cout<<keypoints.size()<<endl;
    //cout<<keypoints2.size()<<endl;
    //cout<<descriptors.size()<<endl;
    //cout<<descriptors<<endl;
    //cout<<descriptors2.size()<<endl;


    //int distance=Match.DescriptorDistance(descriptors,descriptors2);
    //cout<<distance<<endl;
    //特征点可视化;
    drawKeypoints(gray_image,keypoints,img,Scalar(255,0,255),0);

    //drawKeypoints(gray_image2,keypoints2,img2,Scalar(255,0,255),0);
    imshow("image:",img);
    //imshow("image2:",img2);
    //cout<<descriptors.size()<<endl;
    //cout<<keypoints.size()<<endl;
    waitKey(0);
    return 0;
}