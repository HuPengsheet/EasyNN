#include<iostream>
#include<string>
#include"maxpool.h"

namespace easynn{


MaxPool::MaxPool()
{
    one_blob_only=true;
}

void MaxPool::copy_make_border_image(const Mat& input,Mat& input_pad)
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
    input_pad.create(output_w,output_h,input.c);

    for(int i=0;i<input.c;i++)
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

int MaxPool::forward(const Mat& input,Mat& output,const Optional& op)
{
    if(input.dims==1)
    {
        printf("MaxPool do not support 1 dims Mat\n");
    }
    if(ceil_mode||return_indices)
    {
        printf("do not support ceil_mode and return_indices\n");
    }

    int input_h = input.h;
    int input_w = input.w;
    int out_h = (input_h+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1;
    int out_w = (input_h+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1;

    if (input.dims == 2)
        output.create(out_w, out_h);
    else if (input.dims == 3)
        output.create(out_w,out_h,input.c);
    else if (input.dims == 4)
        output.create(out_w,out_h,input.d,input.c);

    
    Mat input_pad=input;
    copy_make_border_image(input,input_pad);
    input_h = input_pad.h;
    input_w = input_pad.w;
    

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

    for(int i=0;i<output.c;i++)
    {
        Mat ptr_in = input_pad.channel(i);
        float* ptr_out = output.channel(i);

        for(int j=0;j<out_h;j++)
        {
            for(int k=0;k<out_w;k++)
            {
                const float* sptr = ptr_in.row(j * stride[1]) + k * stride[0];
                float max = sptr[0];
                for(int m=0;m<kernel_max;m++)
                {
                    if(sptr[kernel_index[m]]>=max)
                        max = sptr[kernel_index[m]];
                }                
                ptr_out[k] = max;
            } 
            ptr_out +=out_w; 
        }
    }
    
    return 0;    
    std::cout<<"MaxPool forward"<<std::endl;
    return 0;

}

int MaxPool::loadParam(std::map<std::string, pnnx::Parameter>& params)
{
    ceil_mode = params["ceil_mode"].b;
    return_indices = params["return_indices"].b;
    padding.assign(params["padding"].ai.begin(),params["padding"].ai.end());     
    dilation.assign(params["dilation"].ai.begin(),params["dilation"].ai.end());    
    kernel_size.assign(params["kernel_size"].ai.begin(),params["kernel_size"].ai.end()); 
    stride.assign(params["stride"].ai.begin(),params["stride"].ai.end());   

    return 0; 
}

}//namespace