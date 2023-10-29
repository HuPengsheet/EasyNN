#ifndef EASYNN_CAT_H
#define EASYNN_CAT_H

#include"layer.h"
#include"mat.h"
namespace easynn{
class Cat:public Layer
{
public:
    Cat();
    virtual int forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op);
};

}//namespace




#endif