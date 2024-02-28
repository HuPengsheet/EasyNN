#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
#include"layers/cxx/silu.h"
#include"optional.h"

TEST(SILU,forward)
{
    easynn::Mat m1(5);
    easynn::Mat m2(3,5);
    easynn::Mat m3(2,3,2);
    std::vector<float> x1={1.3, -1.2, 3.2 ,-5.10 ,10};
    std::vector<std::vector<float>> x2={{-1,2,-2},{-2,-98,100},{3,-4,5},{-31,4,52},{3,-43,59}};
    std::vector<std::vector<std::vector<float>>> x3={{{1,200},{98,-100.3},{4,5}},{{5,-2},{-99,100},{4,51}}};
    
    m1.fillFromArray(x1);
    m2.fillFromArray(x2);
    m3.fillFromArray(x3);

    easynn::Mat m4(5);
    easynn::Mat m5(3,5);
    easynn::Mat m6(2,3,2);
    std::vector<float> x4={1.0216, -0.2778,  3.0747, -0.0309,  9.9995};
    std::vector<std::vector<float>> x5={{-2.6894e-01,  1.7616e+00, -2.3841e-01},
                                        {-2.3841e-01, -0.0000e+00,  1.0000e+02},
                                        { 2.8577e+00, -7.1945e-02,  4.9665e+00},
                                        {-1.0672e-12,  3.9281e+00,  5.2000e+01},
                                        { 2.8577e+00, -9.0951e-18,  5.9000e+01}};
    std::vector<std::vector<std::vector<float>>> x6={{{0.7311,200.000},{98.0000,-0.0000},{3.9281,4.9665}},{{4.9665,-0.23840},{-0.000,100.0000},{3.9281,51.0000}}};
    
    m4.fillFromArray(x4);
    m5.fillFromArray(x5);
    m6.fillFromArray(x6);

    easynn::Silu s1;
    easynn::Mat out_m1;
    easynn::Mat out_m2;
    easynn::Mat out_m3;
    easynn::Optional option;
    s1.forward(m1,out_m1,option);
    s1.forward(m2,out_m2,option);
    s1.forward(m3,out_m3,option);
    EXPECT_EQ(compareMat(out_m1,m4),0);
    EXPECT_EQ(compareMat(out_m2,m5),0);
    EXPECT_EQ(compareMat(out_m3,m6),0);
}