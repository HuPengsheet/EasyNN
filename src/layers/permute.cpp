#include<iostream>
#include"permute.h"
#include"benchmark.h"

namespace easynn{

Permute::Permute()
{
    one_blob_only = true;
}
int Permute::forward(const Mat& input,Mat& output,const Optional& op)
{
    double start = get_current_time();

    if(dims.size()==5 && dims[0]==0)
    {
        dims.erase(dims.begin());
        for(int i=0;i<dims.size();i++)
        {
            dims[i]-=1;
        }
    }
    else if (dims.size()==5)
    {
        printf("do not support 5 dims permute\n");
        return -1;
    }
    int h=input.h;
    int w=input.w;
    int d=input.d;
    int channels=input.c;
    size_t elemsize = input.elemsize;
    output = input.clone();
    if(dims.size()==4)
    {
        output.create(d, w, h, channels, elemsize);
        for (int q = 0; q < channels; q++)
        {
            const Mat m = input.channel(q);
            float* outptr = output.channel(q);

            for (int z = 0; z < h; z++)
            {
                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        *outptr++ = m.depth(j).row(z)[i];
                    }
                }
            }
        }
    }

    else if(dims.size()==3)
    {
        printf("permute not support now\n");
        return -1;
    }
    else if(dims.size()==2)
    {
        printf("permute not support now\n");
        return -1;
    }
    else if(dims.size()==1)
    {
        output = input.clone();
        return 0;
    }
    else 
    {
        printf("permute not support now\n");
        return -1;
    }

    
    double end = get_current_time();
    printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,end-start);
    return 0;
}

int Permute::loadParam(std::map<std::string, pnnx::Parameter>& params)
{

    dims.assign(params["dims"].ai.begin(),params["dims"].ai.end());
    return 0;
}
}//namespace