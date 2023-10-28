#ifndef EASYNN_ADAPTIVEAVGPOOL_H
#define EASYNN_ADAPTIVEAVGPOOL_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class AdaptivePool: public Layer
{
public:
    AdaptivePool();
    virtual int forward(Mat& input,Mat& output);
};

}//namespace

#endif