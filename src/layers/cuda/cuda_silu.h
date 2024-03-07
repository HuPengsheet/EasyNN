#ifndef EASYNN_CUDA_SILU_H
#define EASYNN_CUDA_SILU_H

#include"mat.h"
#include"optional.h"

namespace easynn
{
    void cuda_silu(const Mat& input,Mat& output,const Optional& op);
    void cuda_silu_vec(const Mat& input,Mat& output,const Optional& op);
}



#endif