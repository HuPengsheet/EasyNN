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
    return 0;
}

int Convolution::loadParam(std::map<std::string, pnnx::Parameter>& params)
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

int Convolution::loadBin(std::vector<char>& data)
{
    size_t kernel_max = kernel_size[0]*kernel_size[1];
    size_t data_size = in_channels*kernel_max*out_channels;
    int w= kernel_size[0];
    int h= kernel_size[1];
    int d= in_channels;
    int c= out_channels;
    weight.create(w,h,d,c);
    for()
    return 0;   
}

}//namespace