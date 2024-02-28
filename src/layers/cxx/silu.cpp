#include<iostream>
#include<math.h>
#include"silu.h"
#include"benchmark.h"

#ifdef EASTNN_USE_CUDA
#include"layers/cuda/cuda_silu.h"
#endif
namespace easynn{

Silu::Silu()
{
    one_blob_only = true;
}
int Silu::forward(const Mat& input,Mat& output,const Optional& op)
{
    double start = get_current_time();

#ifdef EASTNN_USE_CUDA
    double cuda_start = get_current_time();
    cuda_silu(input,output,op);
    double cuda_end = get_current_time();
    printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,cuda_end-cuda_start);
    return 0;
#endif

    if (input.dims == 1)
        output.create(input.w);
    else if (input.dims == 2)
        output.create(input.w, input.h);
    else if (input.dims == 3)
        output.create(input.w, input.h, input.c);
    else if (input.dims == 4)
        output.create(input.w, input.h, input.d, input.c);


    for (int q=0; q<input.c; q++)
    {
        float* ptr_input = input.channel(q);
        float* ptr_output = output.channel(q);
        for (int z=0; z<input.d; z++)
        {
            for (int y=0; y<input.h; y++)
            {
                for (int x=0; x<input.w; x++)
                {
                     ptr_output[x] = ptr_input[x]/(1+expf(-ptr_input[x]));   
                }
                ptr_input += input.w;
                ptr_output += output.w;
            }
        }
    }



    double end = get_current_time();
    printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,end-start);

    return 0;
}

}//namespace