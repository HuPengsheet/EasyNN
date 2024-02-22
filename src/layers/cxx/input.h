#ifndef EASYNN_INPUT_H
#define EASYNN_INPUT_H

#include"layer.h"

namespace easynn{


class Input: public Layer
{
public:
    Input();
    virtual int forward();
};

}//namespace

#endif