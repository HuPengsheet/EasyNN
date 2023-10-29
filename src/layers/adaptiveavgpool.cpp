#include<iostream>
#include"adaptiveavgpool.h"
namespace easynn{


AdaptivePool::AdaptivePool()
{
    one_blob_only=true;
}

int AdaptivePool::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();

    std::cout<<"AdaptivePool forward"<<std::endl;
    return 0;
}

}//namespace