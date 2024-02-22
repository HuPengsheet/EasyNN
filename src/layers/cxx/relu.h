#ifndef EASYNN_RELU_H
#define EASYNN_RELU_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class Relu: public Layer
{
public:
    Relu();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
};

}//namespace
#endif