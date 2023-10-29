#include<iostream>
#include"permute.h"


namespace easynn{

Permute::Permute()
{
    one_blob_only = true;
}
int Permute::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();
    std::cout<<"Permute forward"<<std::endl;
    return 0;
}

}//namespace