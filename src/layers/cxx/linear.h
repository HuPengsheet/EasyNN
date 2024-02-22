#ifndef EASYNN_LINEAR_H
#define EASYNN_LINEAR_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class Linear: public Layer
{
public:
    Linear();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
    virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);
    virtual int loadBin(std::map<std::string, pnnx::Attribute>& attrs);

public:

    bool use_bias;
    int in_features;
    int out_features;

    Mat weight;
    Mat bias;
};

}//namespace

#endif