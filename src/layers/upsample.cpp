#include<iostream>
#include"upsample.h"


namespace easynn{

Upsample::Upsample()
{
    one_blob_only = true;
}
int Upsample::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();
    std::cout<<"Upsample forward"<<std::endl;
    return 0;
}

}//namespace