#include<iostream>
#include"silu.h"


namespace easynn{

Silu::Silu()
{
    one_blob_only = true;
}
int Silu::forward(const Mat& input,Mat& output,const Optional& op)
{
    output = input.clone();
    std::cout<<"Silu forward"<<std::endl;
    return 0;
}

}//namespace