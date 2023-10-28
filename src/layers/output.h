#ifndef EASYNN_OUPUT_H
#define EASYNN_OUPUT_H

#include"layer.h"

namespace easynn{


class Output: public Layer
{
public:
    Output();
    virtual int forward();
};

}//namespace

#endif