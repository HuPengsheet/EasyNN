#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
#include"layers/flatten.h"



TEST(FLATTER,forward1)
{

    easynn::Mat m1(3,5);
    std::vector<std::vector<float>> x1={{-1,2,-2},{-2,-98,100},{3,-4,5},{-31,4,52},{3,-43,59}};
    m1.fillFromArray(x1);

    easynn::Mat out(15);
    std::vector<float> x2={-1,2,-2,-2,-98,100,3,-4,5,-31,4,52,3,-43,59};
    out.fillFromArray(x2);

    easynn::Flatten flatten;
    easynn::Optional option;

    easynn::Mat m2;
    flatten.forward(m1,m2,option);
    EXPECT_EQ(compareMat(m2,out),0);
}


TEST(FLATTER,forward2)
{

    easynn::Mat m1(2,3,2);
    std::vector<std::vector<std::vector<float>>> x1={{{1,200},{98,-100.3},{4,5}},{{5,-2},{-99,100},{4,51}}};
    m1.fillFromArray(x1);

    easynn::Mat out(12);
    std::vector<float> x2={1,200,98,-100.3,4,5,5,-2,-99,100,4,51};
    out.fillFromArray(x2);

    easynn::Flatten flatten;
    easynn::Optional option;

    easynn::Mat m2;
    flatten.forward(m1,m2,option);
    EXPECT_EQ(compareMat(m2,out),0);
}