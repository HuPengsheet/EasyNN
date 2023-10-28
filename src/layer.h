#ifndef EASYNN_LAYER_H
#define EASYNN_LAYER_H
#include<vector>
#include<string>
#include"mat.h"
namespace easynn{

class Layer{

public:
    Layer();
    virtual ~Layer();
    virtual int loadParam();
    virtual int loadModel();
    virtual int forward(Mat& input,Mat& output);
    virtual int forward(std::vector<Mat>& input,std::vector<Mat>& output);
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