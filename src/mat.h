#ifndef EASYNN_MAT_H
#define EASYNN_MAT_H

#include<stddef.h>

namespace easynn{

class Mat
{
public:
    //各种构造函数
    Mat();
    Mat(size_t w); 
    Mat(size_t w,size_t h); 
    Mat(size_t w,size_t h,size_t c); 
    ~Mat();

private:
    size_t dims;     //数据的维度 0 or 1 or 2 or 3
    size_t c;           
    size_t h;
    size_t w;
    size_t cstep;
    void* data;       //data的数据  
    int* refcount;   //引用计数的地址 
    
};

}

#endif