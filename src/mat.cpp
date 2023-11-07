#include<stdio.h>
#include<vector>
#include<string.h>
#include"mat.h"
#include"allocator.h"

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

float* Mat::row(int y)
{
    return (float*)((unsigned char*)data + (size_t)w * y * elemsize);
}

float* Mat::row(int y) const
{
    return (float*)((unsigned char*)data + (size_t)w * y * elemsize);
}


Mat Mat::depth(int z)
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize);
}

Mat Mat::depth(int z) const
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize);
}
Mat Mat::channel(int _c)
{
    Mat m(w, h, d, (unsigned char*)data + cstep * _c * elemsize, elemsize);
    m.dims = dims - 1;
    if (dims == 4)
        m.cstep = (size_t)w * h;
    return m;
}

Mat Mat::channel(int _c) const
{
    Mat m(w, h, d, (unsigned char*)data + cstep * _c * elemsize, elemsize);
    m.dims = dims - 1;
    if (dims == 4)
        m.cstep = (size_t)w * h;
    return m;
}

void Mat::fill(int x)
{
    if(isEmpty())
    {
        return ;
    } 
    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);
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
        float *ptr = this->channel(i);
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
    if(dims!=1)
    {
        printf("dims not match\n");
        return ;
    }

    if(isEmpty()||x.empty())
    {
        printf(" vector or mat empty\n");
        return ;
    }
    if(x.size() != w)
    {
        printf(" vector and mat size not match\n");
        return ;
    }
    
    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);;
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
    if(dims!=2)
    {
        printf("dims not match\n");
        return ;
    }

    if(isEmpty()||x.empty())
    {
        printf(" vector or mat empty\n");
        return ;
    }

    if(x.size()!=h|| x[0].size()!=w)
    {
        printf(" vector and mat size not match\n");
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);
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
    if(dims!=3)
    {
        printf("dims not match\n");
        return ;
    }

    if(isEmpty()||x.empty())
    {
        printf(" vector or mat empty\n");
        return ;
    }

    if(x.size()!=c|| x[0].size()!=h || x[0][0].size()!=w)
    {
        printf(" vector and mat size not match\n");
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);
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
void Mat::fillFromArray(std::vector<std::vector<std::vector<std::vector<int>>>> x)
{
    if(dims!=4)
    {
        printf("dims not match\n");
        return ;
    }

    if(isEmpty()||x.empty())
    {
        printf(" vector or mat empty\n");
        return ;
    }

    if(x.size()!=c|| x[0].size()!=d || x[0][0].size()!=w || x[0][0][0].size()!=h)
    {
        printf(" vector and mat size not match\n");
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x[i][z][j][k];
                }
                ptr = ptr+w;
            }
        }
    }   
}

void Mat::fillFromArray(std::vector<float> x)
{
    if(dims!=1)
    {
        printf("dims not match\n");
        return ;
    }

    if(isEmpty()||x.empty())
    {
        printf(" vector or mat empty\n");
        return ;
    }
    if(x.size() != w)
    {
        printf(" vector and mat size not match\n");
        return ;
    }
    
    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);
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
    if(dims!=2)
    {
        printf("dims not match\n");
        return ;
    }

    if(isEmpty()||x.empty())
    {
        printf(" vector or mat empty\n");
        return ;
    }

    if(x.size()!=h|| x[0].size()!=w)
    {
        printf(" vector and mat size not match\n");
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);
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
    if(dims!=3)
    {
        printf("dims not match\n");
        return ;
    }

    if(isEmpty()||x.empty())
    {
        printf(" vector or mat empty\n");
        return ;
    }

    if(x.size()!=c|| x[0].size()!=h || x[0][0].size()!=w)
    {
        printf(" vector and mat size not match\n");
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);
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

void Mat::fillFromArray(std::vector<std::vector<std::vector<std::vector<float>>>> x)
{
    if(dims!=4)
    {
        printf("dims not match\n");
        return ;
    }

    if(isEmpty()||x.empty())
    {
        printf(" vector or mat empty\n");
        return ;
    }

    if(x.size()!=c|| x[0].size()!=d || x[0][0].size()!=h || x[0][0][0].size()!=w)
    {
        printf(" vector and mat size not match\n");
        return ;
    }

    for(int i=0;i<c;i++){
        float *ptr = this->channel(i);
        for(int z=0;z<d;z++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    ptr[k]=x[i][z][j][k];
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

Mat Mat::reshape(int _w) const
{
    if (w * h * d * c != _w)
        return Mat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        Mat m;
        m.create(_w, elemsize);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * d * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.d = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

Mat Mat::reshape(int _w, int _h) const
{
    if (w * h * d * c != _w * _h)
        return Mat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        Mat m;
        m.create(_w, _h, elemsize);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * d * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = 1;

    m.cstep = (size_t)_w * _h;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _c) const
{
    if (w * h * d * c != _w * _h * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize((size_t)_w * _h * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _c, elemsize);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _c);
        return tmp.reshape(_w, _h, _c);
    }

    Mat m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = _c;

    m.cstep = alignSize((size_t)_w * _h * elemsize, 16) / elemsize;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _d, int _c) const
{
    if (w * h * d * c != _w * _h * _d * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h * _d != alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _d, _c, elemsize);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * _d * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * _d * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _d * _c);
        return tmp.reshape(_w, _h, _d, _c);
    }

    Mat m = *this;

    m.dims = 4;
    m.w = _w;
    m.h = _h;
    m.d = _d;
    m.c = _c;

    m.cstep = alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize;
    return m;
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
Mat::Mat(int _w,void* data,size_t _elemsize):dims(1),c(1),d(1),h(1),w(_w),data(data),refcount(0),elemsize(_elemsize)
{
    cstep = _w;
}
Mat::Mat(int _w,int _h,void* data,size_t _elemsize):dims(2),c(1),d(1),h(_h),w(_w),data(data),refcount(0),elemsize(_elemsize)
{
    cstep = _w*_h;
} 
Mat::Mat(int _w,int _h,int _c,void* data,size_t _elemsize):dims(3),c(_c),d(1),h(_h),w(_w),data(data),refcount(0),elemsize(_elemsize)
{
    cstep=alignSize((size_t)w * h * elemsize, 16) / elemsize;
}
Mat::Mat(int _w,int _h,int _d,int _c,void* data,size_t _elemsize):dims(4),c(_c),d(_d),h(_h),w(_w),data(data),refcount(0),elemsize(_elemsize)
{
    cstep = alignSize((size_t)w*h*d*elemsize, 16) / elemsize;
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

float& Mat::operator[](size_t index)
{
    return ((float*)data)[index];
}

float& Mat::operator[](size_t index) const
{
    return ((float*)data)[index];
}

Mat Mat::clone() const
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