#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"

TEST(layer,conv_loadParam)
{
    easynn::Net net;
    EXPECT_EQ(net.loadModel("/home/hupeng/code/github/EasyNN/example/conv.pnnx.param",\
    "/home/hupeng/code/github/EasyNN/example/conv.pnnx.bin"),0);


}