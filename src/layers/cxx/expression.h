#ifndef EASYNN_EXPRESSION_H
#define EASYNN_EXPRESSION_H

#include"layer.h"
#include"mat.h"

namespace easynn{
class Expression:public Layer
{
public:
    Expression();
    virtual int forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op);
    virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);

public:
    std::string expression;
};

}//namespace




#endif