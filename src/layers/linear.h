#ifndef EASYNN_LINEAR_H
#define EASYNN_LINEAR_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class Linear: public Layer
{
public:
    Linear();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
};

}//namespace

#endif