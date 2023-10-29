#include<iostream>
#include"convolution.h"
#include"mat.h"
namespace easynn{


Convolution::Convolution()
{
    one_blob_only=true;
}

int Convolution::forward(const Mat& input,Mat& output,const Optional& op)
{
    output=input.clone();
    std::cout<<"Convolution forward"<<std::endl;
    std::cout<<padding_mode<<std::endl;
    for(auto k:kernel_size)
    {
        std::cout<<k<<std::endl;
    }
    return 0;
}

int Convolution::loadParam(std::map<std::string, pnnx::Parameter> params)
{
    use_bias = params["bias"].b;
    groups = params["groups"].i;
    in_channels = params["in_channels"].i;
    out_channels = params["out_channels"].i;
    padding_mode = params["padding_mode"].s;

    padding.assign(params["padding"].ai.begin(),params["padding"].ai.end());     
    dilation.assign(params["dilation"].ai.begin(),params["dilation"].ai.end());    
    kernel_size.assign(params["kernel_size"].ai.begin(),params["kernel_size"].ai.end()); 
    stride.assign(params["stride"].ai.begin(),params["stride"].ai.end());   
    return 0;   
}

}//namespace