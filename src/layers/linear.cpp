#include<iostream>
#include"linear.h"

namespace easynn{


Linear::Linear()
{
    one_blob_only = true;
}

int Linear::forward(Mat& input,Mat& output)
{
    output = input.clone();
    std::cout<<"Linear forward"<<std::endl;
    return 0;
}

}//namespace