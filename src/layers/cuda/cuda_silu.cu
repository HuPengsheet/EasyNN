#include"nncuda.h"
#include"cuda_silu.h"
#include<stdio.h>


namespace easynn{


__global__ void cuda_silu_forward(float* input,float* output,int n)
{
    CUDA_KERNEL_LOOP(index, n) 
    {   
        //printf("%f     %f\n",input[index],1.0f / (1.0f + expf(input[index])));
        output[index] = input[index] / (1.0f + expf(-input[index]));  
    }

}



void cuda_silu(const Mat& input,Mat& output,const Optional& op)
{
    int count=input.w*input.h*input.c*input.d;

    int nbytes = count*input.elemsize;

    Mat cu_mat_input = input.reshape(count);
    Mat cu_mat_output(count);

    float* d_input,*d_output;

    CHECK(cudaMalloc(&d_input,nbytes));
    CHECK(cudaMalloc(&d_output,nbytes));

    cudaMemcpy(d_input, cu_mat_input.data, nbytes, cudaMemcpyHostToDevice);

    cuda_silu_forward<<<EASYNN_GET_BLOCKS(count), EASYNN_CUDA_NUM_THREADS>>>(d_input,d_output,count);

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

__global__ void cuda_silu_vec_forward(float *input,float*output,size_t n)
{

    int gtid =blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int index=gtid;index<(n+CUDA_VEC_SIZE-1)/CUDA_VEC_SIZE;index+=stride){
        float4 a = reinterpret_cast<float4*>(input)[index];
        float4 c;
        c.x = a.x / (1.0f + expf(-a.x)); 
        c.y = a.y / (1.0f + expf(-a.y));
        c.z = a.z / (1.0f + expf(-a.z));
        c.w = a.w / (1.0f + expf(-a.w));
        reinterpret_cast<float4*>(output)[index] = c;
    }

}

void cuda_silu_vec(const Mat& input,Mat& output,const Optional& op)
{
    int count=input.w*input.h*input.c*input.d;

    int N = (count+CUDA_VEC_SIZE-1)/CUDA_VEC_SIZE*CUDA_VEC_SIZE;

    int nbytes = N*input.elemsize;

    Mat cu_mat_input = input.reshape(count);
    Mat cu_mat_output(count);

    float* d_input,*d_output;

    CHECK(cudaMalloc(&d_input,nbytes));
    CHECK(cudaMalloc(&d_output,nbytes));
    cudaMemset(d_input, 0, nbytes);


    cudaMemcpy(d_input, cu_mat_input.data, count*sizeof(float), cudaMemcpyHostToDevice);

    cuda_silu_vec_forward<<<EASYNN_GET_VEC_BLOCKS(count), EASYNN_CUDA_NUM_THREADS>>>(d_input,d_output,count);

    cudaMemcpy(cu_mat_output.data, d_output, count*sizeof(float), cudaMemcpyDeviceToHost);


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