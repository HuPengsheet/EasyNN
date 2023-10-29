#ifndef EASYNN_UPSAMPLE_H
#define EASYNN_UPSAMPLE_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class Upsample: public Layer
{
public:
    Upsample();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
};

}//namespace
#endif