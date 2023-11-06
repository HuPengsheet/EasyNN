#include<iostream>
#include<string.h>
#include"upsample.h"
#include"benchmark.h"

namespace easynn{

Upsample::Upsample()
{
    one_blob_only = true;
}
int Upsample::forward(const Mat& input,Mat& output,const Optional& op)
{
    if(strcmp(mode.c_str(),"nearest")!=0)
    {
        printf("do not support upsample model %s\n",mode.c_str());
    }
    double start = get_current_time();
    int scale_h = scale_factor[0];
    int scale_w = scale_factor[1];
    int out_h = scale_h*input.h;
    int out_w = scale_w*input.w;

    if (input.dims == 2||input.dims == 1)
        output.create(out_w, out_h);
    else if (input.dims == 3)
        output.create(out_w, out_h, input.c);
    else if (input.dims == 4)
        output.create(out_w, out_h, input.d, input.c);

    for (int q=0; q<input.c; q++)
    {
        float* ptr_input = input.channel(q);
        float* ptr_output = output.channel(q);
        for (int z=0; z<input.d; z++)
        {
            for (int y=0; y<out_h; y++)
            {
                int in_y = std::min((int)(y/scale_h), (input.h - 1));
                for (int x=0; x<out_w; x++)
                {   
                    int in_x = std::min((int)(x/scale_w ), (input.w - 1));
                    *ptr_output++ = ptr_input[in_y * input.w + in_x];   
                }
            }
        }
    }
    
    double end = get_current_time();
    printf("%-15s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,end-start);
    return 0;
}
int Upsample::loadParam(std::map<std::string, pnnx::Parameter>& params)
{
    mode = params["mode"].s;
    scale_factor.assign(params["scale_factor"].af.begin(),params["scale_factor"].af.end());
    return 0;
}

}//namespace