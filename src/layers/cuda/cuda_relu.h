#ifndef EASYNN_CUDA_RELU_H
#define EASYNN_CUDA_RELU_H

#include"mat.h"
#include"optional.h"

namespace easynn
{
    void cuda_relu(const Mat& input,Mat& output,const Optional& op);
}



#endif