#include<iostream>
#include"cat.h"
#include"mat.h"

namespace easynn{
    Cat::Cat()
    {
        one_blob_only = false;
    }
    int Cat::forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op)
    {
        output[0]=input[0].clone();
        std::cout<<"expression forward"<<std::endl;
    }

}//namespace