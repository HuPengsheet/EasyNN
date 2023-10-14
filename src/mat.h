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
    Mat(int _w,size_t _elemsize=4u); 
    Mat(int _w,int _h,size_t _elemsize=4u); 
    Mat(int _w,int _h,int _c,size_t _elemsize=4u); 
    Mat(int _w,int _h,int _d,int _c,size_t _elemsize=4u); 

    void create(int _w,size_t _elemsize=4u); 
    void create(int _w,int _h,size_t _elemsize=4u); 
    void create(int _w,int _h,int _c,size_t _elemsize=4u); 
    void create(int _w,int _h,int _d,int _c,size_t _elemsize=4u); 

    void fill(int x);
    void fill(float x);
    //拷贝构造函数,浅拷贝，refcount加1
    Mat(const Mat& m);

    //运算符号重载
    Mat& operator=(const Mat& m);
    ~Mat();

    void clean();
    void add_ref();
    int isEmpty () const;
    int total() const;
    

    size_t dims;     //数据的维度 0 or 1 or 2 or 3
    int c;     
    int d;      
    int h;
    int w;
    size_t cstep;
    size_t elemsize;
    void* data;       //data的数据  
    int* refcount;   //引用计数的地址 
    
};

}

#endif