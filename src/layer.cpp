#include<layer.h>

namespace easynn{

    Layer::Layer()
    {
        return ;
    }
    Layer::~Layer()
    {
        
    }
    int Layer::loadParam(const std::map<std::string, pnnx::Parameter> params)
    {
        return 0;
    }
    int Layer::loadBin()
    {
        return 0;
    }
    int Layer::forward(const Mat& input,Mat& output,const Optional& op)
    {
        return 0;
    }
    int Layer::forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op)
    {
        return 0;
    }

    int Layer::forwardInplace()
    {
        return 0;
    }

}//namespace