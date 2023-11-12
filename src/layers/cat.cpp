#include<string.h>
#include"cat.h"
#include"mat.h"
#include"benchmark.h"

namespace easynn{

Cat::Cat()
{
    one_blob_only = false;
}
int Cat::forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op)
{
    double start = get_current_time();
    int dims = input[0].dims;
    size_t elemsize = input[0].elemsize;

    if (dims == 3)
    {
    
        int w = input[0].w;
        int h = input[0].h;

        // 总共的channels数
        int top_channels = 0;
        for (size_t b = 0; b < input.size(); b++)
        {
            Mat in = input[b];
            top_channels += in.c;
        }

        Mat& out = output[0];
        out.create(w, h,top_channels, elemsize);

        int q = 0;
        for (size_t b = 0; b < input.size(); b++)
        {
            Mat in = input[b];

            int channels = in.c;
            size_t size = in.cstep * channels;

            unsigned char* ptr = in;
            unsigned char* outptr = out.channel(q);
            memcpy(outptr, ptr, size * elemsize);

            q += channels;
        }
    }
    else
    {
        printf("Cat do not support now %d dims",dims);
    }    

    double end = get_current_time();
    printf("%-25s,cat forward :time=%fms\n",name.c_str(),end-start);
    return 0;
}

int Cat::loadParam(std::map<std::string, pnnx::Parameter>& params)
{
    dim = params["dim"].i;
    dim -=1;
    return 0;
}

}//namespace