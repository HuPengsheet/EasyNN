#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
#include"layers/upsample.h"
#include"optional.h"




TEST(UPSAMPLE,forward)
{
    easynn::Mat m1(5);
    easynn::Mat m2(3,5);
    easynn::Mat m3(2,3,2);
    std::vector<float> x1={1.0216, -0.2778,  3.0747, -0.0309,  9.9995};
    std::vector<std::vector<float>> x2={{-1,2,-2},{-2,-98,100},{3,-4,5},{-31,4,52},{3,-43,59}};
    std::vector<std::vector<std::vector<float>>> x3={{{1,200},{98,-100.3},{4,5}},{{5,-2},{-99,100},{4,51}}};
    
    m1.fillFromArray(x1);
    m2.fillFromArray(x2);
    m3.fillFromArray(x3);

    easynn::Mat m4(10,2);
    easynn::Mat m5(6,10);
    easynn::Mat m6(4,6,2);
    std::vector<std::vector<float>> x4=\
                        {{ 1.0216,  1.0216, -0.2778, -0.2778,  3.0747,  3.0747, -0.0309,
                        -0.0309,  9.9995,  9.9995},
                        { 1.0216,  1.0216, -0.2778, -0.2778,  3.0747,  3.0747, -0.0309,
                        -0.0309,  9.9995,  9.9995}};
    std::vector<std::vector<float>> x5=\
                                        {{ -1.,  -1.,   2.,   2.,  -2.,  -2.},
                                            { -1.,  -1.,   2.,   2.,  -2.,  -2.},
                                            { -2.,  -2., -98., -98., 100., 100.},
                                            { -2.,  -2., -98., -98., 100., 100.},
                                            {  3.,   3.,  -4.,  -4.,   5.,   5.},
                                            {  3.,   3.,  -4.,  -4.,   5.,   5.},
                                            {-31., -31.,   4.,   4.,  52.,  52.},
                                            {-31., -31.,   4.,   4.,  52.,  52.},
                                            {  3.,   3., -43., -43.,  59.,  59.},
                                            {  3.,   3., -43., -43.,  59.,  59.}};
    std::vector<std::vector<std::vector<float>>> x6=\
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
    
    m4.fillFromArray(x4);
    m5.fillFromArray(x5);
    m6.fillFromArray(x6);

    easynn::Upsample upsample;
    upsample.mode = "nearest";
    upsample.scale_factor.push_back(2.0);
    upsample.scale_factor.push_back(2.0);


    easynn::Mat out_m1;
    easynn::Mat out_m2;
    easynn::Mat out_m3;
    easynn::Optional option;
    upsample.forward(m1,out_m1,option);
    upsample.forward(m2,out_m2,option);
    upsample.forward(m3,out_m3,option);

    EXPECT_EQ(compareMat(out_m1,m4),0);
    EXPECT_EQ(compareMat(out_m2,m5),0);
    EXPECT_EQ(compareMat(out_m3,m6),0);


}


template<typename T>
std::vector<size_t> get_nested_vector_sizes(const std::vector<T>& vec) {
    std::vector<size_t> sizes;
    sizes.push_back(vec.size());

    for (const auto& item : vec) {
        if (std::is_same_v<T, std::vector<typename T::value_type>>) {
            auto sub_sizes = get_nested_vector_sizes(item);
            sizes.insert(sizes.end(), sub_sizes.begin(), sub_sizes.end());
        }
    }

    return sizes;
}