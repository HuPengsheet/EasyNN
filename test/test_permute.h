#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
#include"layers/permute.h"
#include"optional.h"


std::vector<std::vector<std::vector<std::vector<float>>>> permute_input = \
        {{{{0.3765, 0.6601},
           {0.8344, 0.1654}},

          {{0.8061, 0.5652},
           {0.6344, 0.8243}},

          {{0.5888, 0.7781},
           {0.2078, 0.1379}},

          {{0.5917, 0.8670},
           {0.6892, 0.8355}}},


         {{{0.0059, 0.9083},
           {0.9211, 0.1498}},

          {{0.3458, 0.3369},
           {0.7829, 0.9449}},

          {{0.7218, 0.4120},
           {0.7404, 0.4953}},

          {{0.5888, 0.9852},
           {0.4225, 0.4163}}},


         {{{0.0247, 0.7075},
           {0.2999, 0.9630}},

          {{0.8158, 0.0463},
           {0.3084, 0.3792}},

          {{0.8941, 0.9431},
           {0.2986, 0.4835}},

          {{0.6188, 0.7671},
           {0.8713, 0.2998}}}};

std::vector<std::vector<std::vector<std::vector<float>>>> permute_output = \
        {{{{0.3765, 0.8061, 0.5888, 0.5917},
           {0.6601, 0.5652, 0.7781, 0.8670}},

          {{0.8344, 0.6344, 0.2078, 0.6892},
           {0.1654, 0.8243, 0.1379, 0.8355}}},


         {{{0.0059, 0.3458, 0.7218, 0.5888},
           {0.9083, 0.3369, 0.4120, 0.9852}},

          {{0.9211, 0.7829, 0.7404, 0.4225},
           {0.1498, 0.9449, 0.4953, 0.4163}}},


         {{{0.0247, 0.8158, 0.8941, 0.6188},
           {0.7075, 0.0463, 0.9431, 0.7671}},

          {{0.2999, 0.3084, 0.2986, 0.8713},
           {0.9630, 0.3792, 0.4835, 0.2998}}}};

TEST(PERMUTE,forward)
{
    easynn::Mat input(2,2,4,3);
    easynn::Mat output(4,2,2,3);
    input.fillFromArray(permute_input);
    output.fillFromArray(permute_output);

    easynn::Permute permute;
    permute.dims.push_back(0);
    permute.dims.push_back(1);
    permute.dims.push_back(3);
    permute.dims.push_back(4);
    permute.dims.push_back(2);
    easynn::Optional op;
    
    easynn::Mat m1;
    permute.forward(input,m1,op);
    printMat(m1);
    printMat(output);
    EXPECT_EQ(compareMat(m1,output),0);
}