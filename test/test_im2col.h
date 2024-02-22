#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
#include"layers/convolution.h"

std::vector<std::vector<std::vector<float>>> im2col_input1=\
                                                {{{   1.0000,    1.0000,  200.0000,  200.0000},
                                                    {   1.0000,    1.0000,  200.0000,  200.0000},
                                                    {  98.0000,   98.0000, -100.3000, -100.3000},
                                                    {  98.0000,   98.0000, -100.3000, -100.3000},
                                                    {   4.0000,    4.0000,    5.0000,    5.0000},
                                                    {   4.0000,    4.0000,    5.0000,    5.0000}},

                                                    {{   5.0000,    5.0000,   -2.0000,   -2.0000},
                                                    {   5.0000,    5.0000,   -2.0000,   -2.0000},
                                                    { -99.0000,  -99.0000,  100.0000,  100.0000},
                                                    { -99.0000,  -99.0000,  100.0000,  100.0000},
                                                    {   4.0000,    4.0000,   51.0000,   51.0000},
                                                    {   4.0000,    4.0000,   51.0000,   51.0000}}};

TEST(CONV,im2col)
{
    easynn::Mat input1(4,6,2);
    input1.fillFromArray(im2col_input1);

    easynn::Optional op;

    std::vector<int> stride;
    stride.push_back(1);
    stride.push_back(1);

    std::vector<int> kernel_size;
    kernel_size.push_back(2);
    kernel_size.push_back(2);

    std::vector<int> dilation;
    dilation.push_back(1);
    dilation.push_back(1);



    easynn::Mat out;
    im2col(input1,out,op,kernel_size,stride,dilation);

    printMat(out);

}

