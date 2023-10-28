#include<iostream>
#include"flatten.h"

namespace easynn{


Flatten::Flatten()
{
    one_blob_only = true;
}

int Flatten::forward(Mat& input,Mat& output)
{
    output = input.clone();
    std::cout<<"Flatten forward"<<std::endl;
    return 0;
}

}//namespace