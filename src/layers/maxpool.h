#ifndef EASYNN_MAXPOOL_H
#define EASYNN_MAXPOOL_H

#include"layer.h"

namespace easynn{


class MaxPool: public Layer
{
public:
    MaxPool();
    virtual int forward();
};

}//namespace

#endif