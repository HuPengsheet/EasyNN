#ifndef EASYNN_LAYER_H
#define EASYNN_LAYER_H
#include<vector>
#include<string>
#include<map>
#include"mat.h"
#include"optional.h"
#include"ir.h"
namespace easynn{

class Layer{

public:
    Layer();
    virtual ~Layer();
    virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);
    virtual int loadBin(std::map<std::string, pnnx::Attribute>& attrs);
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
    virtual int forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op);
    virtual int forwardInplace();

public:

    // one input and one output blob
    bool one_blob_only;

public:
    // custom user data
    void* userdata;
    // layer type index
    int typeindex;

    // layer type name
    std::string type;
    // layer name
    std::string name;

    // blob index which this layer needs as input
    std::vector<int> bottoms;
    // blob index which this layer produces as output
    std::vector<int> tops;
    // shape hint
    // std::vector<Mat> bottom_shapes;
    // std::vector<Mat> top_shapes;
};

}



#endif