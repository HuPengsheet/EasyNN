#ifndef EASYNN_LAYER_H
#define EASYNN_LAYER_H
#include<vector>
#include<string>

namespace easynn{

class Layer{

public:
    Layer();
    virtual ~Layer();
    virtual int loadParam();
    virtual int loadModel();
    virtual int forward();
    virtual int forwardInplace();

public:

    // one input and one output blob
    bool one_blob_only;
    // support inplace inference
    bool support_inplace;
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
    std::vector<Mat> bottom_shapes;
    std::vector<Mat> top_shapes;
};

}



#endif