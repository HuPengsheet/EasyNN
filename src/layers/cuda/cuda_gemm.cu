#include"nncuda.h"
#include"cuda_gemm.h"
#include<stdio.h>


namespace easynn{


 template <int BLOCK>
__global__ void cuda_sgemm_forward(size_t m,size_t n,size_t k,float* a,float* b,float* c,float *d_bias)
{
    
  int _m = blockIdx.x * BLOCK + threadIdx.x;
    int _n = blockIdx.y * BLOCK + threadIdx.y;
    if (_m < m and _n < n) {
      float sum = 0.f;
      for (int i = 0; i < k; ++i) {
        sum += a[_m * k + i] * b[i * n + _n];
      }
      c[_m * n + _n] = sum+d_bias[_m];
    }

}


//a和b分别是两个以及im2col的矩阵
void cuda_gemm(const Mat& input_a,const Mat& input_b,Mat& output_c,const Mat& bias,const Optional& op)
{
    if(input_a.w!=input_b.h) 
    {
        printf("input_a.w!=input_b.h , can not mutl\n");
    }

    //printf("input_a.w=k=%d  input_a.h=m=%d  input_a.c=%d\n",input_a.w,input_a.h,input_a.c);
    //printf("input_b.w=n=%d  input_b.h=k=%d  input_b.c=%d\n",input_b.w,input_b.h,input_b.c);

    int m = input_a.h;
    int k = input_a.w;
    int n = input_b.w;
    
    //printf("%d \n",bias.w);

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
    CHECK(cudaMemcpy(d_b,(float *)input_b.data,b_nbytes,cudaMemcpyHostToDevice));
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
}



}// namespace