#include<iostream>
#include<stdio.h>
#include<string.h>
#include"convolution.h"
#include"mat.h"

namespace easynn{

Convolution::Convolution()
{
    one_blob_only=true;
}

void Convolution::copy_make_border_image(const Mat& input,Mat& input_pad)
{
    int padding_h  = padding[0];
    int padding_w = padding[1];
    int input_w = input.w;
    int input_h = input.h;
    int output_w = input_w+2*padding_w;
    int output_h = input_h+2*padding_h;
    if(padding_h==0 && padding_w==0)
    {
        input_pad = input;
        return ;
    }
    input_pad.create(output_w,output_h,in_channels);

    for(int i=0;i<in_channels;i++)
    {
        float * input_ptr = input.channel(i);
        float * pad_ptr = input_pad.channel(i);
        int j=0;
        //padding top
        for(;j<padding_h;j++)
        {
            for(int k=0;k<output_w;k++)
            {
                pad_ptr[k]=0;
            }
            pad_ptr +=output_w;
        }

        //padding centor
        for(;j<output_h-padding_h;j++)
        {
            int k=0;
            for(;k<padding_w;k++)
            {
                pad_ptr[k]=0;
            }
            for(;k<output_w-padding_w;k++)
            {
                pad_ptr[k]=input_ptr[k-padding_w];
            }
           for(;k<output_w;k++)
            {
                pad_ptr[k]=0;
            }            
            input_ptr += input_w;
            pad_ptr +=output_w;
            
        }

        //padding bottom
        for(;j<output_h;j++)
        {
            for(int k=0;k<output_w;k++)
            {
                pad_ptr[k]=0;
            }
            pad_ptr +=output_w;
        }
    }

}

int Convolution::forward(const Mat& input,Mat& output,const Optional& op)
{   

    int input_h = input.h;
    int input_w = input.w;
    int out_h = (input_h+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1;
    int out_w = (input_h+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1;
    output.create(out_w,out_h,out_channels);
    
    Mat input_pad=input;
    if(strcmp(padding_mode.c_str(), "zeros")==0)
    {   
        copy_make_border_image(input,input_pad);
        input_h = input_pad.h;
        input_w = input_pad.w;
    }
    else
    {
        printf("do not support padding mode %s\n",padding_mode.c_str());
        return -1;
    }
    
    printf("in_channels:%d, out_channels:%d, input_h:%d ,input_w:%d ,out_h%d ,out_w%d\n",in_channels,out_channels,input.h,input.w,out_h,out_w);


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

    for(int i=0;i<out_channels;i++)
    {
        float* ptr = output.channel(i);

        for(int j=0;j<out_h;j++)
        {
            for(int k=0;k<out_w;k++)
            {
                float sum=0.f;
                if(use_bias)
                    sum=bias[i];

                float* kptr =weight.channel(i);
                for(int q = 0; q < in_channels; q++)
                {
                    const Mat m = input_pad.channel(q);
                    const float* sptr = m.row(j * stride[1]) + k * stride[0];

                    for(int m=0;m<kernel_max;m++)
                    {
                        float val = sptr[kernel_index[m]];
                        float wt = kptr[m];
                        sum +=val*wt;
                    }

                    kptr += kernel_max;
                }

                ptr[k] = sum;
            } 
            ptr +=out_w;
        }
    }
    
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