//
// Created by true on 18-11-17.
//验证矩阵中的某些操作,验证其输出
//验证map数据
//由于是opencv2.4.9 编译它的编译器是旧版本,c++11 则用的是G++ 5因此在c++_namespace中使用opencv不可行;
#include <iostream>
#include <opencv2/opencv.hpp>
#include <map>
#include <string>


using namespace std;
using namespace cv;

int main()
{

    //设有点40个,内点率0.5;若要拥有8个内点, 则需要迭代多少次呢?
    //计算8次均选到内点的概率;
    float m=pow(0.8,8);
    cout<<log(0.01)/log(1-m);
    return 0;
}





