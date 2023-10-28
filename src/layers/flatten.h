#ifndef EASYNN_FLATTEN_H
#define EASYNN_FLATTEN_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class Flatten: public Layer
{
public:
    Flatten();
    virtual int forward(Mat& input,Mat& output);
};

}//namespace

#endif