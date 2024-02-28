#ifndef EASYNN_SILU_H
#define EASYNN_SILU_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class Silu: public Layer
{
public:
    Silu();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
};

}//namespace
#endif