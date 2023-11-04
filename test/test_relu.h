#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
#include"layers/relu.h"
#include"optional.h"
TEST(Relu,forward)
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
    std::vector<float> x4={1.3 , 0 , 3.2 , 0 , 10};
    std::vector<std::vector<float>> x5={{0,2,0},{0,-0,100},{3,0,5},{0,4,52},{3,0,59}};
    std::vector<std::vector<std::vector<float>>> x6={{{1,200},{98,0},{4,5}},{{5,0},{0,100},{4,51}}};
    
    m4.fillFromArray(x4);
    m5.fillFromArray(x5);
    m6.fillFromArray(x6);

    easynn::Relu r1;
    easynn::Mat out_m1;
    easynn::Mat out_m2;
    easynn::Mat out_m3;
    easynn::Optional option;
    r1.forward(m1,out_m1,option);
    r1.forward(m2,out_m2,option);
    r1.forward(m3,out_m3,option);
    printMat(out_m3);
    printMat(m6);
    EXPECT_EQ(compareMat(out_m1,m4),0);
    EXPECT_EQ(compareMat(out_m2,m5),0);
    EXPECT_EQ(compareMat(out_m3,m6),0);
}