#ifndef EASYNN_ADAPTIVEAVGPOOL_H
#define EASYNN_ADAPTIVEAVGPOOL_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class AdaptivePool: public Layer
{
public:
    AdaptivePool();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
    virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);

    std::vector<int> output_size;
};

}//namespace

#endif