#include"contiguous.h"
#include"benchmark.h"
namespace easynn{


Contiguous::Contiguous()
{
    one_blob_only=true;
}

int Contiguous::forward(const Mat& input,Mat& output,const Optional& op)
{
    double start = get_current_time();
    
    output = input.clone();

    double end = get_current_time();
    printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,end-start);

    return 0;
}

}//namespace