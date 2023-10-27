#ifndef EASYNN_RELU_H
#define EASYNN_RELU_H

#include"layer.h"

namespace easynn{


class Relu: public Layer
{
public:
    Relu();
    virtual int forward();
};

}//namespace
#endif