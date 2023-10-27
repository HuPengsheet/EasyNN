#ifndef EASYNN_EXPRESSION_H
#define EASYNN_EXPRESSION_H

#include"layer.h"

namespace easynn{
class Expression:public Layer
{
public:
    Expression();
    virtual int forward();
};

}//namespace




#endif