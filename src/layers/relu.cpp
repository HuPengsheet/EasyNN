#include"iostream"
#include"relu.h"


namespace easynn{

Relu::Relu()
{
    one_blob_only = true;
}
int Relu::forward(Mat& input,Mat& output)
{
    output = input.clone();
    std::cout<<"Relu forward"<<std::endl;
    return 0;
}

}//namespace