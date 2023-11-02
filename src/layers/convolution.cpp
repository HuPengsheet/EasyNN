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
    
    int input_h = input.h;
    int input_w = input.w;
    int out_h = (input_h+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1;
    int out_w = (input_h+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1;
    output.create(out_w,out_h,out_channels);
    
    size_t kernel_max = kernel_size[0]*kernel_size[1];
    std::vector<int> kernel_index(kernel_max);
    {
        int gap = input_w * dilation[1] - kernel_size[1] * dilation[0];
        int p=0;
        int q=0;
        for(int i=0;i<kernel_size[0];i++)
        {
            for(int j=0;j<kernel_size[1];j++)
            {
                kernel_index[p] = q;
                p++;
                q+=dilation[1];
            }
            q +=gap;
        }
    }

    
    // for(auto index:kernel_index) std::cout<<index<<std::endl;
    // std::cout<<out_h<<"    "<<out_w<<std::endl;
    
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

int Convolution::loadBin(std::map<std::string, pnnx::Attribute>& attrs)
{   

    float* weight_data = (float*)(&attrs["weight"].data[0]);
    size_t kernel_max = kernel_size[0]*kernel_size[1];
    size_t kernel_channels = kernel_size[0]*kernel_size[1]*in_channels;
    size_t data_size = in_channels*kernel_max*out_channels;
    int w= kernel_size[0];
    int h= kernel_size[1];
    int d= in_channels;
    int c= out_channels;
    weight.create(w,h,d,c);
    for(int i=0;i<out_channels;i++)
    {
        float* ptr=weight.channel(i);
        for(int j=0;j<in_channels;j++)
        {
            for(int k=0;k<h;k++)
            {
               for(int m=0;m<w;m++)
               {
                ptr[m]=weight_data[i*kernel_channels+j*kernel_max+k*h+m];
               }
                ptr = ptr+w;
            }
        }
    }


    if(use_bias)
    {
        float* weight_data = (float*)(&attrs["bias"].data[0]);
        bias.create(out_channels);
        for(int i=0;i<out_channels;i++)
        {
            bias[i]=weight_data[i];
        }
    }
    return 0;   
}

}//namespace