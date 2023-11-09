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

    bool one_blob_only;  //�������Ƿ�ֻ��һ������

public:

    std::string type;   //���ӵ�����
    std::string name;   //���ڵ�����

    std::vector<int> bottoms;   //�����ӵ�����blob������
    std::vector<int> tops;      //�����ӵ����blob������

};

}



#endif