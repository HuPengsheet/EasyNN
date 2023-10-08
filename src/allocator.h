#ifndef EASYNN_ALLOCATOR_H
#define EASYNN_ALLOCATOR_H

#include <stdlib.h>


#define EASYNN_MALLOC_ALIGN 64
#define EASYNN_MALLOC_OVERREAD 64

static  size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

//fastMalloc反对的是对其的64字节地址，同时也会额外分配64字节的空间
//额外分配的空间,一方面是为了安全，一方面是我们在使用的时候会溢出，比如mat的引用计数
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



#endif