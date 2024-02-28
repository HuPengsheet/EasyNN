#ifndef EASYNN_PERMUTE_H
#define EASYNN_PERMUTE_H


#include"layer.h"
#include"mat.h"

namespace easynn{


class Permute: public Layer
{
public:
    Permute();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
    virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);
public:
    std::vector<int> dims;

};

}//namespace
#endif