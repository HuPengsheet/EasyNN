#include <stdio.h>

#ifdef EASTNN_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include"nncuda.h"

namespace easynn{
    


void* fastCudaMalloc(size_t size)
{
    void *ptr = 0;
    CHECK(cudaMalloc((void **)&ptr, size + EASYNN_MALLOC_OVERREAD));
    return ptr;
}

void fastCudaFree(void * ptr)
{
    CHECK(cudaFree(ptr));
}


#endif   //EASTNN_USE_CUDA



}//namespace