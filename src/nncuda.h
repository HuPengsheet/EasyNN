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


namespace easynn{

int getGpuNum();
void* fastCudaMalloc(size_t size);
void  fastCudaFree(void * ptr);

} //namespace easynn


#endif  //EASTNN_USE_CUDA

#endif  //EASYNN_CUDA_H