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

public:

    bool one_blob_only;  //该算子是否只有一个输入

public:

    std::string type;   //算子的类型
    std::string name;   //算在的名字

    std::vector<int> bottoms;   //该算子的输入blob的索引
    std::vector<int> tops;      //该算子的输出blob的索引

};

}



#endif