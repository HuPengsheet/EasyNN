#include"nncuda.h"
#include"cuda_linear.h"
#include"cuda_gemm.h"
#include<stdio.h>


namespace easynn{


template <int BLOCK>
__global__ void cuda_sgemm_forward(size_t m,size_t n,size_t k,float* a,float* b,float* c,float *bias)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int global_idx= bx*BLOCK+tx;
    int global_idy= by*BLOCK+ty;

    float *begin_a = a + bx * BLOCK * k;
    float *end_a = begin_a + k;
    float *a_bottom = a+(m-1)*k;
    float *a_block_bottom = begin_a+(BLOCK-1)*k;
    int a_x_gap = (a_bottom>=a_block_bottom) ? BLOCK : (BLOCK-(a_block_bottom-a_bottom)/k);


    float *begin_b = b + by * BLOCK;
    float *end_b = b+(k-1)*n;
    float *b_right = b+n;
    float *b_block_right = begin_b+BLOCK;
    int b_y_gap = (b_right>=b_block_right) ? BLOCK : (BLOCK-(b_block_right-b_right));
        

    float sum = 0.f;
    for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;a_ptr += BLOCK, b_ptr += BLOCK * n) 
    {

        __shared__ float ashare[BLOCK][BLOCK];
        __shared__ float bshare[BLOCK][BLOCK];

        float* a_block_right = a_ptr+BLOCK;
        int a_y_gap = (end_a>=a_block_right) ? BLOCK : (BLOCK-(a_block_right-end_a));

        float* b_block_bottom = b_ptr+(BLOCK-1) * n;
        int b_x_gap = (end_b>=b_block_bottom) ? BLOCK : (BLOCK-(b_block_bottom-end_b)/n);


        if(tx<a_x_gap&&ty<a_y_gap) ashare[tx][ty] = a_ptr[tx * k + ty];
        if(tx<b_x_gap&&ty<b_y_gap) bshare[tx][ty] = b_ptr[tx * n + ty];

        __syncthreads();



        #pragma unroll
        for (int kk = 0; kk < BLOCK; ++kk) 
        {
            sum += ashare[tx][kk] * bshare[kk][ty];
        }

        __syncthreads();

        ashare[tx][ty]=0;
        bshare[tx][ty]=0; 
        __syncthreads();


    }

    if(global_idx<m&&global_idy<n)
    {
        c[global_idx*n+global_idy] = sum+bias[global_idx];
    }

}


//a和b分别是两个以及im2col的矩阵
void cuda_linear(const Mat& input_a,const Mat& input_b,Mat& output_c,const Mat& bias,const Optional& op)
{

    Mat cuda_b = input_b.reshape(1,input_b.w);

    if(input_a.w!=cuda_b.h) 
    {
        printf("input_a.w!=input_b.h , can not mutl\n");
    }

    // printf("input_a.w=k=%d  input_a.h=m=%d  input_a.c=%d\n",input_a.w,input_a.h,input_a.c);
    // printf("cuda_b.w=n=%d  cuda_b.h=k=%d  cuda_b.c=%d\n",cuda_b.w,cuda_b.h,cuda_b.c);

    int m = input_a.h;
    int k = input_a.w;
    int n = cuda_b.w;
    

    output_c.create(m,n);

    float *d_a,*d_b,*d_c,*d_bias;
    size_t a_nbytes = m*k*sizeof(float);
    size_t b_nbytes = n*k*sizeof(float);
    size_t c_nbytes = m*n*sizeof(float);
    size_t d_nbytes = m*sizeof(float);

    CHECK(cudaMalloc(&d_a,a_nbytes));
    CHECK(cudaMalloc(&d_b,b_nbytes));
    CHECK(cudaMalloc(&d_c,c_nbytes));
    CHECK(cudaMalloc(&d_bias,d_nbytes));

    //printf("11 %p %p %p %p \n",d_a,d_b,d_c,d_bias);

    CHECK(cudaMemcpy(d_a,(float *)input_a.data,a_nbytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,(float *)cuda_b.data,b_nbytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias,(float *)bias.data,d_nbytes,cudaMemcpyHostToDevice));

    constexpr int BLOCK = 16;
    //constexpr int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK,(n + BLOCK - 1) / BLOCK);
    cuda_sgemm_forward<BLOCK><<<grid,block>>>(m,n,k,d_a,d_b,d_c,d_bias);
    cudaDeviceSynchronize();

    cudaMemcpy(output_c.data,d_c,c_nbytes,cudaMemcpyDeviceToHost);
    //printf("22 %p %p %p %p \n",d_a,d_b,d_c,d_bias);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFree(d_bias));

    int count=output_c.w*output_c.h*output_c.c*output_c.d;
    output_c.reshape(count);
}





}// namespace