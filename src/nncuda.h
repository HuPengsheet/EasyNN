#ifndef EASYNN_NNCUDA_H
#define EASYNN_NNCUDA_H

#ifdef EASTNN_USE_CUDA

#define EASYNN_MALLOC_ALIGN 64
#define EASYNN_MALLOC_OVERREAD 64

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int EASYNN_CUDA_NUM_THREADS = 256;
const int CUDA_VEC_SIZE =4;

// CUDA: number of blocks for threads.
inline int EASYNN_GET_BLOCKS(const int N) {
  return (N + EASYNN_CUDA_NUM_THREADS - 1) / EASYNN_CUDA_NUM_THREADS;
}

inline int EASYNN_GET_VEC_BLOCKS(const int N) {
  return ((N + EASYNN_CUDA_NUM_THREADS - 1) / EASYNN_CUDA_NUM_THREADS+CUDA_VEC_SIZE-1)/CUDA_VEC_SIZE;
}


namespace easynn{

} //namespace easynn


#endif  //EASTNN_USE_CUDA

#endif  //EASYNN_CUDA_H