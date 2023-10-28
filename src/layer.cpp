#include<layer.h>

namespace easynn{

    Layer::Layer()
    {
        return ;
    }
    Layer::~Layer()
    {
        
    }
    int Layer::loadParam()
    {
        return 0;
    }
    int Layer::loadModel()
    {
        return 0;
    }
    int Layer::forward(Mat& input,Mat& output)
    {
        return 0;
    }
    int Layer::forward(std::vector<Mat>& input,std::vector<Mat>& output)
    {
        return 0;
    }

    int Layer::forwardInplace()
    {
        return 0;
    }

}//namespace