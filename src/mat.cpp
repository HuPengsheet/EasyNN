#include"mat.h"
#include"allocator.h"
#include<stdio.h>

namespace easynn{

    int Mat::is_empty () const
    {
        if(w<0)
            return 1;
        return 0;
    }
    int Mat::total() const
    {
        return cstep*c * elemsize;
    }

    void Mat::add_ref()
    {
        if(refcount)
        {
            *refcount+=1;
        }
    }


    void Mat::create(int _w,size_t _elemsize)
    {
        dims=1;
        w=_w;
        h=1;
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
        c=_w;
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

    Mat::~Mat()
    {

    }
    Mat::Mat():dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
    {
        
    }
    
    Mat::Mat(int w,size_t _elemsize):dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
    {
        Mat::create(w,_elemsize);
    }
    Mat::Mat(int w,int h,size_t _elemsize):dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
    {
        Mat::create(w,h,_elemsize);
    }
    Mat::Mat(int w,int h,int c,size_t _elemsize):dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
    {
        Mat::create(w,h,c,_elemsize);
    }

    Mat::Mat(const Mat& m):dims(m.dims),c(m.c),h(m.h),w(m.w),cstep(m.cstep),data(m.data),refcount(m.refcount),elemsize(m.elemsize)
    {
        add_ref();
    }

}