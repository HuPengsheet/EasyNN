#include<iostream>
#include"expression.h"
#include"mat.h"
#include"benchmark.h"
namespace easynn{

Expression::Expression()
{
    one_blob_only = false;
}

int Expression::forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op)
{
    double start = get_current_time();
    Mat add_left = input[0];
    Mat add_right = input[1];
    
    if(add_left.w!=add_right.w || add_left.h!=add_right.h || add_left.c!=add_right.c || add_left.d!=add_right.d)
    {
        printf("the shape is different, can not add \n");
        return -1;
    }

    Mat& out = output[0];
    int w=add_left.w;
    int h=add_left.h;
    int d=add_left.d;
    int c=add_left.c;

    if(add_left.dims == 1)
        out.create(w);
    if (add_left.dims == 2)
        out.create(w, h);
    else if (add_left.dims == 3)
        out.create(w,h,c);
    else if (add_left.dims == 4)
        out.create(w,h,d,c);

    for (int q=0; q<c; q++)
    {
        float* left_ptr = add_left.channel(q);
        float* right_output = add_right.channel(q);
        float* out_ptr = out.channel(q);

        for (int z=0; z<d; z++)
        {
            for (int y=0; y<h; y++)
            {
                for (int x=0; x<w; x++)
                {
                    out_ptr[x]=left_ptr[x]+right_output[x];
                }
                left_ptr += w;
                right_output += w;
                out_ptr+=w;
            }
        }
    }

    double end = get_current_time();
    printf("%-15s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),c,c,h,w,h,w,end-start);
    return 0;
}

int Expression::loadParam(std::map<std::string, pnnx::Parameter>& params)
{
    expression=params["expr"].s;
    return 0;
}
        
    

}//namespace