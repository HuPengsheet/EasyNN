#include<iostream>
#include"relu.h"


namespace easynn{

Relu::Relu()
{
    one_blob_only = true;
}
int Relu::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();
    std::cout<<"Relu forward"<<std::endl;
    return 0;
}

}//namespace