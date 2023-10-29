#ifndef EASYNN_MAXPOOL_H
#define EASYNN_MAXPOOL_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class MaxPool: public Layer
{
public:
    MaxPool();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
};

}//namespace

#endif