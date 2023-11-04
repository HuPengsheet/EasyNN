#include<iostream>
#include"relu.h"


namespace easynn{

Relu::Relu()
{
    one_blob_only = true;
}
int Relu::forward(const Mat& input,Mat& output,const Optional& op)
{
    
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
                    if(ptr_input[x]<0)
                        ptr_output[x]=0;
                    else
                        ptr_output[x] = ptr_input[x];   
                }
                ptr_input += input.w;
                ptr_output += output.w;
            }
        }
    }
    return 0;
}


}//namespace