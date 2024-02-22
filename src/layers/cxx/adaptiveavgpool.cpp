#include"adaptiveavgpool.h"
#include"benchmark.h"

namespace easynn{
    
AdaptivePool::AdaptivePool()
{
    one_blob_only=true;
}
int AdaptivePool::forward(const Mat& input,Mat& output,const Optional& op)
{
    double start = get_current_time();
    int input_h = input.h;
    int input_w = input.w;
    int out_h = output_size[0];
    int out_w = output_size[1];

    if (input.dims == 2)
        output.create(out_w, out_h);
    else if (input.dims == 3)
        output.create(out_w,out_h,input.c);
    else if (input.dims == 4)
        output.create(out_w,out_h,input.d,input.c);
        
    for(int i=0;i<output.c;i++)
    {
        float* ptr_in = input.channel(i);
        float* ptr_out = output.channel(i);

        for(int j=0;j<out_h;j++)
        {
            // floor div
            const int ih0 = input_h * j / out_h;
            // ceil div
            const int ih1 = (input_h * (j + 1) + out_h - 1) / out_h;
            const int hk = ih1 - ih0; 

            for(int k=0;k<out_w;k++)
            {
                const int iw0 = input_w * k / out_w;
                // ceil div
                const int iw1 = (input_w * (k + 1) + out_w - 1) / out_w;
                const int wk = iw1 - iw0;

                float sum = 0;
                for (int ih = ih0; ih < ih1; ih++)
                {
                    for (int iw = iw0; iw < iw1; iw++)
                    {
                        sum += ptr_in[ih * input_w + iw];
                    }
                }              
                ptr_out[k] = sum / hk / wk;
            } 
            ptr_out +=out_w;
        }
    }
    double end = get_current_time();
    printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,out_h,out_w,end-start);
    return 0;
}

// int AdaptivePool::forward(const Mat& input,Mat& output,const Optional& op)
// {
//     int input_h = input.h;
//     int input_w = input.w;
//     int out_h = output_size[0];
//     int out_w = output_size[1];

//     if (input.dims == 2)
//         output.create(out_w, out_h);
//     else if (input.dims == 3)
//         output.create(out_w,out_h,input.c);
//     else if (input.dims == 4)
//         output.create(out_w,out_h,input.d,input.c);

//     int stride_h = input_h / out_h;
//     int stride_w = input_w / out_w;

//     int pooling_h = input_h - (out_h - 1) * stride_h;
//     int pooling_w = input_w - (out_w - 1) * stride_w;

//     size_t kernel_max = pooling_h*pooling_w;
//     std::vector<int> kernel_index(kernel_max);
//     {
//         int gap = input_w-pooling_w; 
//         int p=0;
//         int q=0;
//         for(int i=0;i<pooling_h;i++)
//         {
//             for(int j=0;j<pooling_w;j++)
//             {
//                 kernel_index[p] = q;
//                 p++;
//                 q++;
//             }
//             q +=gap;
//         }
//     }

//     for(int i=0;i<output.c;i++)
//     {
//         Mat ptr_in = input.channel(i);
//         float* ptr_out = output.channel(i);

//         for(int j=0;j<out_h;j++)
//         {
//             for(int k=0;k<out_w;k++)
//             {
//                 const float* sptr = ptr_in.row(j * stride_h) + k * stride_w;
//                 float sum = 0;
//                 for(int m=0;m<kernel_max;m++)
//                 {
//                     sum += sptr[kernel_index[m]];
//                 }                
//                 ptr_out[k] = sum/kernel_max;
//             } 
//             ptr_out +=out_w;
//         }
//     }

//     std::cout<<"AdaptivePool forward"<<std::endl;
//     return 0;
// }

int AdaptivePool::loadParam(std::map<std::string, pnnx::Parameter>& params)
{
    output_size.assign(params["output_size"].ai.begin(),params["output_size"].ai.end());  
    return 0;
}

}//namespace