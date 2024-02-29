#include"nncuda.h"
#include"cuda_relu.h"
#include<stdio.h>


namespace easynn{


__global__ void cuda_relu_forward(float* input,float* output,int n)
{
    CUDA_KERNEL_LOOP(index, n) 
    {   
        output[index] = fmaxf(0.0f,input[index]);  
    }

}



void cuda_relu(const Mat& input,Mat& output,const Optional& op)
{
    int count=input.w*input.h*input.c*input.d;

    int nbytes = count*input.elemsize;

    Mat cu_mat_input = input.reshape(count);
    Mat cu_mat_output(count);

    float* d_input,*d_output;

    CHECK(cudaMalloc(&d_input,nbytes));
    CHECK(cudaMalloc(&d_output,nbytes));

    cudaMemcpy(d_input, cu_mat_input.data, nbytes, cudaMemcpyHostToDevice);

    cuda_relu_forward<<<CAFFE_GET_BLOCKS(count), EASYNN_CUDA_NUM_THREADS>>>(d_input,d_output,count);

    cudaMemcpy(cu_mat_output.data, d_output, nbytes, cudaMemcpyDeviceToHost);


    if (input.dims == 1)
        output=cu_mat_output.reshape(input.w);
    else if (input.dims == 2)
        output=cu_mat_output.reshape(input.w, input.h);
    else if (input.dims == 3)
        output=cu_mat_output.reshape(input.w, input.h, input.c);
    else if (input.dims == 4)
        output=cu_mat_output.reshape(input.w, input.h, input.d, input.c);

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
        
}

}// namespace