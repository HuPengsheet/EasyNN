#ifndef EASYNN_MAT_H
#define EASYNN_MAT_H

#include<stddef.h>
#include<vector>
namespace easynn{

class Mat
{
public:

    Mat();
    Mat(int _w,size_t _elemsize=4u); 
    Mat(int _w,int _h,size_t _elemsize=4u); 
    Mat(int _w,int _h,int _c,size_t _elemsize=4u); 
    Mat(int _w,int _h,int _d,int _c,size_t _elemsize=4u); 
    Mat(const Mat& m);

    Mat(int _w,void* data,size_t _elemsize=4u); 
    Mat(int _w,int _h,void* data,size_t _elemsize=4u); 
    Mat(int _w,int _h,int _c,void* data,size_t _elemsize=4u); 
    Mat(int _w,int _h,int _d,int _c,void* data,size_t _elemsize=4u); 


    Mat& operator=(const Mat& m);
    float& operator[](size_t index); 
    float& operator[](size_t index) const; 

    template<typename T>
    operator T*();
    template<typename T>
    operator const T*() const;
 
    void create(int _w,size_t _elemsize=4u); 
    void create(int _w,int _h,size_t _elemsize=4u); 
    void create(int _w,int _h,int _c,size_t _elemsize=4u); 
    void create(int _w,int _h,int _d,int _c,size_t _elemsize=4u); 

    void fill(int x);
    void fill(float x);

    Mat reshape(int _w) const;
    Mat reshape(int _w, int _h) const;
    Mat reshape(int _w, int _h, int _c) const;
    Mat reshape(int _w, int _h, int _d, int _c) const;

    void fillFromArray(std::vector<int> x);
    void fillFromArray(std::vector<std::vector<int>> x);
    void fillFromArray(std::vector<std::vector<std::vector<int>>> x);
    void fillFromArray(std::vector<std::vector<std::vector<std::vector<int>>>> x);
    void fillFromArray(std::vector<float> x);
    void fillFromArray(std::vector<std::vector<float>> x);
    void fillFromArray(std::vector<std::vector<std::vector<float>>> x);
    void fillFromArray(std::vector<std::vector<std::vector<std::vector<float>>>> x);
    
    ~Mat();
    
    void clean();
    void add_ref();

    float* row(int y);
    float* row(int y) const;
    Mat depth(int z);
    Mat depth(int z) const;
    Mat channel(int _c);
    Mat channel(int _c) const ;
    
    int isEmpty () const;
    int total() const;
    Mat clone() const;
    

    size_t dims;     //数据的维度 0 or 1 or 2 or 3 or 4
    int c;     
    int d;      
    int h;
    int w;
    size_t cstep;
    size_t elemsize;
    void* data;       //data的数据  
    int* refcount;   //引用计数的地址 
    
};

template<typename T>
Mat::operator T*()
{
    return (T*)data;
}

template<typename T>
Mat::operator const T*() const
{
    return (T*)data;
}

} //namespace




#endif //EASYNN_MAT_H