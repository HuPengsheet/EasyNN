#ifndef EASYNN_MAT_H
#define EASYNN_MAT_H

#include<stddef.h>

namespace easynn{

class Mat
{
public:
    //各种构造函数
    //

    //普通构造函数
    Mat();
    Mat(size_t _w,size_t _elemsize=4u); 
    Mat(size_t _w,size_t _h,size_t _elemsize=4u); 
    Mat(size_t _w,size_t _h,size_t _c,size_t _elemsize=4u); 

    void create(size_t _w,size_t _elemsize=4u); 
    void create(size_t _w,size_t _h,size_t _elemsize=4u); 
    void create(size_t _w,size_t _h,size_t _c,size_t _elemsize=4u); 

    //拷贝构造函数
    Mat(const Mat& m);

    //运算重载
    Mat& operator=(const Mat& m);
    ~Mat();

private:
    size_t dims;     //数据的维度 0 or 1 or 2 or 3
    size_t c;           
    size_t h;
    size_t w;
    size_t cstep;
    size_t elemsize;
    void* data;       //data的数据  
    int* refcount;   //引用计数的地址 
    
};

}

#endif