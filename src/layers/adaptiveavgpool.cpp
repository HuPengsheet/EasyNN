#include<iostream>
#include"adaptiveavgpool.h"
namespace easynn{


AdaptivePool::AdaptivePool()
{
    one_blob_only=true;
}

int AdaptivePool::forward(Mat& input,Mat& output)
{
    output = input.clone();

    std::cout<<"AdaptivePool forward"<<std::endl;
    return 0;
}

}//namespace