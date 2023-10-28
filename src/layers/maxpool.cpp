#include<iostream>
#include"maxpool.h"

namespace easynn{


MaxPool::MaxPool()
{
    one_blob_only=true;
}

int MaxPool::forward(Mat& input,Mat& output)
{
    output = input.clone();
    std::cout<<"MaxPool forward"<<std::endl;
    return 0;
}

}//namespace