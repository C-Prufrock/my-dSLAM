/*
// Created by true on 18-12-3.
** Author: Xiaoyun Lu
 * @brief
 * 验证动态环境中为何Initialize不能正常初始化
 * 排除下列因素的干扰
 * 1.缺纹理,因此匹配的特征点不足,难以解算F,H模型
 * 2.图像模糊
 *
*/



#include<iostream>

#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;
int main()
{

    cv::Mat matSrc = cv::imread("/home/true/CLionProjects/VslamBasedStaticDescribeSet/Vocabulary/rgbd_dataset_freiburg3_walking_rpy/rgb/1341846647.734820LDOF.ppm");
    Mat matRGB[3];
    split(matSrc, matRGB);
    int Channels[] = { 0 };
    int nHistSize[] = { 256 };
    float range[] = { 0, 255 };
    const float* fHistRanges[] = { range };
    Mat histR, histG, histB;
    // 计算直方图
    calcHist(&matRGB[0], 1, Channels, Mat(), histB, 1, nHistSize, fHistRanges, true, false);
    calcHist(&matRGB[1], 1, Channels, Mat(), histG, 1, nHistSize, fHistRanges, true, false);
    calcHist(&matRGB[2], 1, Channels, Mat(), histR, 1, nHistSize, fHistRanges, true, false);

    // 创建直方图画布
    int nHistWidth = 800;
    int nHistHeight = 600;
    int nBinWidth = cvRound((double)nHistWidth / nHistSize[0]);
    Mat matHistImage(nHistHeight, nHistWidth, CV_8UC3, Scalar(255, 255, 255));
    // 直方图归一化
    normalize(histB, histB, 0.0, matHistImage.rows, NORM_MINMAX, -1, Mat());
    normalize(histG, histG, 0.0, matHistImage.rows, NORM_MINMAX, -1, Mat());
    normalize(histR, histR, 0.0, matHistImage.rows, NORM_MINMAX, -1, Mat());
    // 在直方图中画出直方图
    for (int i = 1; i < nHistSize[0]; i++)
    {
        line(matHistImage,
             Point(nBinWidth * (i - 1), nHistHeight - cvRound(histB.at<float>(i - 1))),
             Point(nBinWidth * (i), nHistHeight - cvRound(histB.at<float>(i))),
             Scalar(255, 0, 0),
             2,
             8,
             0);
        line(matHistImage,
             Point(nBinWidth * (i - 1), nHistHeight - cvRound(histG.at<float>(i - 1))),
             Point(nBinWidth * (i), nHistHeight - cvRound(histG.at<float>(i))),
             Scalar(0, 255, 0),
             2,
             8,
             0);
        line(matHistImage,
             Point(nBinWidth * (i - 1), nHistHeight - cvRound(histR.at<float>(i - 1))),
             Point(nBinWidth * (i), nHistHeight - cvRound(histR.at<float>(i))),
             Scalar(0, 0, 255),
             2,
             8,
             0);
    }
    // 显示直方图
    imshow("histogram", matHistImage);
    //imwrite(strPath + "histogram.jpg", matHistImage);
    waitKey();
    return 0;

}

