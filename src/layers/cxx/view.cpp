#include<iostream>
#include"view.h"
#include"benchmark.h"

namespace easynn{

View::View()
{
    one_blob_only = true;
}

int View::forward(const Mat& input,Mat& output,const Optional& op)
{
    
    double start = get_current_time();
    if(shape.size()==5&&shape[0]==1)
    {
	    shape.erase(shape.begin());
    }
    else if(shape.size()==5)
    {
        printf("view not support shape\n");
        return -1;
    }

    int index =-1;
    for(int i=0;i<shape.size();i++)
    {
        if(shape[i]==-1)
        {
            index = i;
        }
    }
    if(index!=-1)
    {
        int s=1;
        for(int i=0;i<shape.size();i++)
        {
            if(i!=index)
                s *=shape[i];
        }  
        shape[index] = input.w*input.h*input.d*input.c/s;
    }
    output = input.clone();
    if(shape.size()==4)
    {
        output=output.reshape(shape[3],shape[2],shape[1],shape[0]);
    }
    else if(shape.size()==3)
    {
        output=output.reshape(shape[2],shape[1],shape[0]);
    }
    else if(shape.size()==2)
    {
        output=output.reshape(shape[1],shape[0]);
    }
    else if(shape.size()==1)
    {
        output=output.reshape(shape[0]);
    }
    else 
    {
        return -1;
    }
    double end = get_current_time();
    printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d ,time=%fms\n",name.c_str(),input.c,output.c,input.h,input.w,output.h,output.w,end-start);
    return 0;
}

int View::loadParam(std::map<std::string, pnnx::Parameter>& params)
{
    shape.assign(params["shape"].ai.begin(),params["shape"].ai.end());
    return 0;
}
}//namespace