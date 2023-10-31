#ifndef EASYNN_CONVOLUTION_H
#define EASYNN_CONVOLUTION_H

#include<vector>
#include<map>
#include"layer.h"
#include"mat.h"
namespace easynn{


class Convolution: public Layer
{
public:
    Convolution();

    virtual int forward(const Mat& input,Mat& output,const Optional& op);
    virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);
    virtual int loadBin(std::vector<char>& data);
public:

    bool use_bias;        //type1
    
    int groups;       //type 2  
    int in_channels;  //trpe 2
    int out_channels; //type 2

    std::string padding_mode;     //type 4

    std::vector<int> padding;     //type 5
    std::vector<int> dilation;    //type 5
    std::vector<int> kernel_size; //type 5
    std::vector<int> stride;      //type 5
    

    Mat weight;
    Mat bias;
};


} //namespace

#endif