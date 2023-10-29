#include<iostream>
#include"flatten.h"

namespace easynn{


Flatten::Flatten()
{
    one_blob_only = true;
}

int Flatten::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();
    std::cout<<"Flatten forward"<<std::endl;
    return 0;
}

}//namespace