#ifndef EASYNN_MAXPOOL_H
#define EASYNN_MAXPOOL_H

#include"layer.h"
#include"mat.h"
namespace easynn{


class MaxPool: public Layer
{
public:
    MaxPool();
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
    virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);
    void copy_make_border_image(const Mat& input,Mat& input_pad);

public:
    bool ceil_mode ;
    bool return_indices;
    std::vector<int> padding;     //type 5
    std::vector<int> dilation;    //type 5
    std::vector<int> kernel_size; //type 5
    std::vector<int> stride;      //type 5
    
};

}//namespace

#endif