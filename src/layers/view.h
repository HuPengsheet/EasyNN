#ifndef EASYNN_VIEW_H
#define EASYNN_VIEW_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class View: public Layer
{
public:
    View();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
};

}//namespace
#endif