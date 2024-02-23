#ifndef EASYNN_ALLOCATOR_H
#define EASYNN_ALLOCATOR_H

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "nncuda.h"

#define EASYNN_MALLOC_ALIGN 64
#define EASYNN_MALLOC_OVERREAD 64

static  size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}


static void* fastMalloc(size_t size)
{   
    void * ptr = 0;
    if (posix_memalign(&ptr, EASYNN_MALLOC_ALIGN, size + EASYNN_MALLOC_OVERREAD))
        ptr = 0;
    return ptr;
}

static  void fastFree(void* ptr)
{
    if (ptr)
    {
        free(ptr);
    }
}

#ifdef EASTNN_USE_CUDA

static void* fastCudaMalloc(size_t size)
{
    void *ptr = 0;
    check(cudaMalloc((void **)&ptr, size + EASYNN_MALLOC_OVERREAD));
    return ptr;
}

static void fastCudaFree(void * ptr)
{
    void *ptr = 0;
    check(cudaFree(cudaFree));
}

#endif


#endif