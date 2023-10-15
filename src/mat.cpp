#include"mat.h"
#include"allocator.h"
#include<stdio.h>

namespace easynn{


void Mat::clean()
{   
    if(refcount && (*refcount-=1)==1)
    {
        fastFree(data);
    }

}

int Mat::isEmpty () const
{
    return data == 0 || total() == 0;
}
int Mat::total() const
{
    return cstep*c*elemsize;
}

void Mat::add_ref()
{   
    if(refcount)
    {
        *refcount=*refcount+1;
    }
}

void Mat::fill(int x)
{
    if(isEmpty())
    {
        return ;
    } 
    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x;
                }
                ptr = ptr+w;
            }
        }
    }   
}

void Mat::fill(float x)
{
    if(isEmpty())
    {
        return ;
    } 
    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x;
                }
                ptr = ptr+w;
            }
        }
    }   
}
void Mat::fillFromArray(int * x)
{
    
    if(dims!=1)
    {
        return ;
    }
    if(sizeof(x)/sizeof(x[0])!=w)
    {   
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    printf("%d\n",x[k]);
                    ptr[k]=x[k];
                }
                ptr = ptr+w;
            }
        }
    }   
    
}
// void fillFromArray(int **);
// void fillFromArray(int ***);
// void fillFromArray(float *);
// void fillFromArray(float **);
// void fillFromArray(float ***);
void Mat::create(int _w,size_t _elemsize)
{
    dims=1;
    w=_w;
    h=1;
    d=1;
    c=1;
    cstep = _w;
    elemsize = _elemsize;
    size_t totalsize = alignSize(total(), 4);

    if(totalsize>0)
    {
        data = fastMalloc(totalsize);
    }
    else
    {
        printf("totalsize <0\n");
    }
    if(data)
    {
        refcount=(int*)((unsigned char *)data+totalsize);
        *refcount=1;
    }
}

void Mat::create(int _w,int _h,size_t _elemsize)
{
    dims=2;
    w=_w;
    h=_h;
    d=1;
    c=1;
    cstep = _w*_h;
    elemsize = _elemsize;
    size_t totalsize = alignSize(total(), 4);

    if(totalsize>0)
    {
        data = fastMalloc(totalsize);
    }
    else
    {
        printf("totalsize <0\n");
    }
    if(data)
    {
        refcount=(int*)((unsigned char *)data+totalsize);
        *refcount=1;
    }          

}
void Mat::create(int _w,int _h,int _c,size_t _elemsize)
{   
    dims=3;
    w=_w;
    h=_h;
    d=1;
    c=_c;
    elemsize = _elemsize;
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
    size_t totalsize = alignSize(total(), 4);
    
    if(totalsize>0)
    {
        data = fastMalloc(totalsize);
    }
    else
    {
        printf("totalsize <0\n");
    }
    if(data)
    {
        refcount=(int*)((unsigned char *)data+totalsize);
        *refcount=1;
    }         
}

void Mat::create(int _w,int _h,int _d,int _c,size_t _elemsize)
{   
    dims=4;
    w=_w;
    h=_h;
    d=_d;
    c=_c;
    elemsize = _elemsize;
    cstep = alignSize((size_t)w*h*d*elemsize, 16) / elemsize;
    size_t totalsize = alignSize(total(), 4);
    
    if(totalsize>0)
    {
        data = fastMalloc(totalsize);
    }
    else
    {
        printf("totalsize <0\n");
    }
    if(data)
    {
        refcount=(int*)((unsigned char *)data+totalsize);
        *refcount=1;
    }         
}

Mat::~Mat()
{
    clean();
}
Mat::Mat():dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
{
    
}

Mat::Mat(int _w,size_t _elemsize):dims(0),c(0),d(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
{
    Mat::create(_w,_elemsize);
}
Mat::Mat(int _w,int _h,size_t _elemsize):dims(0),c(0),d(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
{
    Mat::create(_w,_h,_elemsize);
}
Mat::Mat(int _w,int _h,int _c,size_t _elemsize):dims(0),c(0),d(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
{
    Mat::create(_w,_h,_c,_elemsize);
}
Mat::Mat(int _w,int _h,int _d,int _c,size_t _elemsize):dims(0),c(0),d(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
{
    Mat::create(_w,_h,_d,_c,_elemsize);
}
Mat::Mat(const Mat& m):dims(m.dims),c(m.c),h(m.h),w(m.w),cstep(m.cstep),data(m.data),refcount(m.refcount),elemsize(m.elemsize)
{
    add_ref();
}



}