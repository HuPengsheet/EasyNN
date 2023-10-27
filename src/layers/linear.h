#ifndef EASYNN_LINEAR_H
#define EASYNN_LINEAR_H

#include"layer.h"

namespace easynn{


class Linear: public Layer
{
public:
    Linear();
    virtual int forward();
};

}//namespace

#endif