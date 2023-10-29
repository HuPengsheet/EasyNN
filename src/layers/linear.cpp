#include<iostream>
#include"linear.h"

namespace easynn{


Linear::Linear()
{
    one_blob_only = true;
}

int Linear::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();
    std::cout<<"Linear forward"<<std::endl;
    return 0;
}

}//namespace