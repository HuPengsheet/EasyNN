#ifndef EASYNN_CUDA_GEMM_H
#define EASYNN_CUDA_GEMM_H

#include"mat.h"
#include"optional.h"

namespace easynn
{
    void cuda_gemm(const Mat& input_a,const Mat& input_b,Mat& output_c,const Mat& bias,const Optional& op);
}



#endif