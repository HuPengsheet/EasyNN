#include<iostream>
#include<string.h>
#include"flatten.h"
#include"benchmark.h"
namespace easynn{


Flatten::Flatten()
{
    one_blob_only = true;
}

int Flatten::forward(const Mat& input,Mat& output,const Optional& op)
{
    double start = get_current_time();
    int w = input.w;
    int h = input.h;
    int d = input.d;
    int channels = input.c;
    size_t elemsize = input.elemsize;
    int size = w * h * d;

    output.create(size * channels);


    for (int q = 0; q < channels; q++)
    {
        const unsigned char* ptr = input.channel(q);
        unsigned char* outptr = (unsigned char*)output + size * elemsize * q;
        memcpy(outptr, ptr, size * elemsize);
    }
    double end = get_current_time();
    printf("%-15s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,end-start);
    return 0;
}

}//namespace