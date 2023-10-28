#ifndef EASYNN_LINEAR_H
#define EASYNN_LINEAR_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class Linear: public Layer
{
public:
    Linear();
    virtual int forward(Mat& input,Mat& output);
};

}//namespace

#endif