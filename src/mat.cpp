#include"mat.h"
#include"allocator.h"
#include<stdio.h>
#include<vector>
#include<string.h>
namespace easynn{


void Mat::clean()
{   
    if(refcount && (*refcount-=1)==0)
    {
        fastFree(data);
    }
    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;

    cstep = 0;
    data = 0;
    refcount = 0;

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

void Mat::fillFromArray(std::vector<int> x)
{
    if(dims!=1||isEmpty())
    {
        return ;
    }
    if(x.size() != w)
    {
        return ;
    }
    
    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x[k];
                }
                ptr = ptr+w;
            }
        }
    }  
}

void Mat::fillFromArray(std::vector<std::vector<int>> x)
{
    if(dims!=2||isEmpty())
    {
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x[j][k];
                }
                ptr = ptr+w;
            }
        }
    }    
}

void Mat::fillFromArray(std::vector<std::vector<std::vector<int>>> x)
{
    if(dims!=3||isEmpty())
    {
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x[i][j][k];
                }
                ptr = ptr+w;
            }
        }
    }   
}

void Mat::fillFromArray(std::vector<float> x)
{
    if(dims!=1||isEmpty())
    {
        return ;
    }
    if(x.size() != w)
    {
        return ;
    }
    
    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x[k];
                }
                ptr = ptr+w;
            }
        }
    }  
}

void Mat::fillFromArray(std::vector<std::vector<float>> x)
{
    if(dims!=2||isEmpty())
    {
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x[j][k];
                }
                ptr = ptr+w;
            }
        }
    }    
}

void Mat::fillFromArray(std::vector<std::vector<std::vector<float>>> x)
{
    if(dims!=3||isEmpty())
    {
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = (float *)((char *)data+cstep*i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x[i][j][k];
                }
                ptr = ptr+w;
            }
        }
    }   
}

void Mat::create(int _w,size_t _elemsize)
{
    clean();
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
    clean();
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
    clean();
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
    clean();
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
Mat::Mat():dims(0),c(0),d(0),h(0),w(0),cstep(0),data(0),refcount(0),elemsize(0)
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
Mat::Mat(const Mat& m):dims(m.dims),c(m.c),d(m.d),h(m.h),w(m.w),cstep(m.cstep),data(m.data),refcount(m.refcount),elemsize(m.elemsize)
{
    add_ref();
}

Mat& Mat::operator=(const Mat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        *m.refcount= *m.refcount+1;

    clean();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;

    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

Mat Mat::clone()
{
    if (isEmpty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize);
    else if (dims == 2)
        m.create(w, h, elemsize);
    else if (dims == 3)
        m.create(w, h, c, elemsize);
    else if (dims == 4)
        m.create(w, h, d, c, elemsize);

    if (total() > 0)
    {
        memcpy(m.data, data, total());
    }
    return m;
}

}