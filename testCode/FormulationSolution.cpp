//
// Created by true on 18-11-19.
//
#include<iostream>
#include<math.h>

using namespace std;

int main(){
    double a= 29;
    double b=-14;
    double c=-5;
    double extremeX;
    extremeX=(-0.5)*b/a;
    cout<<extremeX<<endl;
    double result;
    result = a*pow(extremeX,2)+b*extremeX+c;
    cout<<"result : "<<result<<endl;
    return 0;
}

