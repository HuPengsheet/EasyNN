#ifndef EASYNN_ADAPTIVEAVGPOOL_H
#define EASYNN_ADAPTIVEAVGPOOL_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class AdaptivePool: public Layer
{
public:
    AdaptivePool();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
};

}//namespace

#endif