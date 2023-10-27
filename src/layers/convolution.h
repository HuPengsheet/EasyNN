#ifndef EASYNN_CONVOLUTION_H
#define EASYNN_CONVOLUTION_H

#include"layer.h"
namespace easynn{


class Convolution: public Layer
{
public:
    Convolution();

    virtual int forward();
};


} //namespace

#endif