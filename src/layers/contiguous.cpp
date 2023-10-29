#include<iostream>
#include"contiguous.h"
namespace easynn{


Contiguous::Contiguous()
{
    one_blob_only=true;
}

int Contiguous::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();

    std::cout<<"Contiguous forward"<<std::endl;
    return 0;
}

}//namespace