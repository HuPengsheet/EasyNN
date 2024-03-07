#include<iostream>
#include"linear.h"
#include"string.h"
#include"benchmark.h"

#ifdef EASTNN_USE_CUDA
#include"layers/cuda/cuda_linear.h"
#endif

namespace easynn{

Linear::Linear()
{
    one_blob_only = true;
}

int Linear::forward(const Mat& input,Mat& output,const Optional& op)
{

    
    if(input.dims!=1)
    {
        printf("do not support 2 or 3 or 4 dims Mat for Linear\n");
        return -1;
    }

    #ifdef EASTNN_USE_CUDA
        double cuda_start = get_current_time();
        cuda_linear(weight,input,output,bias,op);
        double cuda_end = get_current_time();
        printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,cuda_end-cuda_start);
        return 0;
    #endif
    
    double start = get_current_time();
    output.create(out_features);

    for(int i=0;i<out_features;i++)
    {   
        float sum=0;
        if(use_bias)
            sum = bias[i];
        for(int j=0;j<in_features;j++)
        {
            sum+=weight[i*in_features+j]*input[j];
        }
        output[i] = sum;
    }

    double end = get_current_time();
    printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,end-start);
    return 0;
}

int Linear::loadParam(std::map<std::string, pnnx::Parameter>& params)
{   
    use_bias = params["bias"].b;
    in_features = params["in_features"].i;
    out_features = params["out_features"].i;

    return 0;   
}

int Linear::loadBin(std::map<std::string, pnnx::Attribute>& attrs)
{   

    void* weight_data = (void*)(&attrs["weight"].data[0]);
    weight.create(in_features,out_features);
    memcpy((void *)weight,weight_data,in_features*out_features*weight.elemsize);


    if(use_bias)
    {
        void* bias_data = (void*)(&attrs["bias"].data[0]);
        bias.create(out_features);
        memcpy((void *)bias,bias_data,out_features*bias.elemsize);
    }
    return 0;   
}
}//namespace