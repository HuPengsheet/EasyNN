#include"mat.h"
#include"allocator.h"
#include<stdio.h>

namespace easynn{

    
    void Mat::create(size_t _w,size_t _elemsize)
    {
        dims=1;
        w=_w;
        h=1;
        c=1;
        cstep = _w;
        elemsize = _elemsize;
        size_t totalsize = alignSize(cstep * elemsize, 4);

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
            refcount=(int*)(int(data)+totalsize);
            *refcount=1;
        }
            


    }
    void Mat::create(size_t _w,size_t _h,size_t _elemsize)
    {
        dims=2;
        w=_w;
        h=_h;
        c=1;
        cstep = _w*_h;
        elemsize = _elemsize;
        size_t totalsize = alignSize(cstep * elemsize, 4);

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
            refcount=(int*)(int(data)+totalsize);
            *refcount=1;
        }          

    }
    void Mat::create(size_t _w,size_t _h,size_t _c,size_t _elemsize)
    {   
        dims=3;
        w=_w;
        h=_h;
        c=_w;
        elemsize = _elemsize;
        cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
        size_t totalsize = alignSize(cstep*c * elemsize, 4);
        
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
            refcount=(int*)(int(data)+totalsize);
            *refcount=1;
        }         
    }


    Mat::Mat():dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
    {
        
    }
    
    Mat::Mat(size_t w,size_t _elemsize):dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
    {
        Mat::create(w,_elemsize);
    }
    Mat::Mat(size_t w,size_t h,size_t _elemsize):dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
    {
        Mat::create(w,h,_elemsize);
    }
    Mat::Mat(size_t w,size_t h,size_t c,size_t _elemsize):dims(0),c(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
    {
        Mat::create(w,h,c,_elemsize);
    }


}