#include<iostream>
#include"convolution.h"
#include"mat.h"
namespace easynn{


Convolution::Convolution()
{
    one_blob_only=true;
}

int Convolution::forward(Mat& input,Mat& output)
{
    output=input.clone();
    std::cout<<"Convolution forward"<<std::endl;

    return 0;
}

}//namespace