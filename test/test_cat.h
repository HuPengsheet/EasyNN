#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
#include"layers/cxx/cat.h"

std::vector<std::vector<std::vector<float>>> cat_input1=\
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

std::vector<std::vector<std::vector<float>>> cat_input2=\
                                                {{{   2.0000,    2.0000,  200.0000,  200.0000},
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

std::vector<std::vector<std::vector<float>>> cat_output=\
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
                                                    {   4.0000,    4.0000,   51.0000,   51.0000}},

                                                    {{   2.0000,    2.0000,  200.0000,  200.0000},
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



TEST(CAT,forward)
{

    easynn::Mat input1(4,6,2);
    easynn::Mat input2(4,6,2);
    easynn::Mat output(4,6,4);

    input1.fillFromArray(cat_input1);
    input2.fillFromArray(cat_input2);
    output.fillFromArray(cat_output);

    easynn::Optional op;
    easynn::Cat cat;
    cat.dim = 0;

    std::vector<easynn::Mat> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);

    std::vector<easynn::Mat> outputs;
    easynn::Mat m1;
    outputs.push_back(m1);


    cat.forward(inputs,outputs,op);
    EXPECT_EQ(compareMat(outputs[0],output),0);
    

}