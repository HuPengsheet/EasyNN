#ifndef EASYNN_ADAPTIVEAVGPOOL_H
#define EASYNN_ADAPTIVEAVGPOOL_H

#include"layer.h"

namespace easynn{


class AdaptivePool: public Layer
{
public:
    AdaptivePool();
    virtual int forward();
};

}//namespace

#endif