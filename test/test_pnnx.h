#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"

TEST(pnnx,loadmodel)
{
    pnnx::Graph net;
    EXPECT_EQ(net.load("/home/hupeng/code/github/EasyNN/example/res18.pnnx.param",\
    "/home/hupeng/code/github/EasyNN/example/res18.pnnx.bin"),0);
    printf("  \n");

}