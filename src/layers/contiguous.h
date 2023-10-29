#ifndef EASYNN_CONTIGUOUS_H
#define EASYNN_CONTIGUOUS_H

#include"layer.h"
#include"mat.h"
namespace easynn{
class Contiguous:public Layer
{
public:
    Contiguous();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
};

}//namespace




#endif