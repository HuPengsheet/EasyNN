#include<iostream>
#include"expression.h"
#include"mat.h"

namespace easynn{
    Expression::Expression()
    {
        one_blob_only = false;
    }
    int Expression::forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op)
    {
        output[0]=input[0].clone();
        std::cout<<"expression forward"<<std::endl;
    }

}//namespace