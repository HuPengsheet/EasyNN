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
    virtual int loadBin(std::map<std::string, pnnx::Attribute>& attrs);
    void copy_make_border_image(const Mat& input, Mat& input_pad);
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

void im2col(const Mat & input,Mat& output,const Optional& opt,const std::vector<int> kernel_size,const std::vector<int> stride,const std::vector<int> dilation);


} //namespace

#endif