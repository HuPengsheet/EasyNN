#ifndef EASYNN_CONVOLUTION_H
#define EASYNN_CONVOLUTION_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class Convolution: public Layer
{
public:
    Convolution();

    virtual int forward(Mat& input,Mat& output);
};


} //namespace

#endif