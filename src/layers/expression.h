#ifndef EASYNN_EXPRESSION_H
#define EASYNN_EXPRESSION_H

#include"layer.h"
#include"mat.h"
namespace easynn{
class Expression:public Layer
{
public:
    Expression();
    virtual int forward(std::vector<Mat>& input,std::vector<Mat>& output);
};

}//namespace




#endif