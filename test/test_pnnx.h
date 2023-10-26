
#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"

TEST(net,printLayer)
{
    easynn::Net net;
    EXPECT_EQ(net.loadParam("/home/hupeng/code/github/EasyNN/example/res18.pnnx.param",\
    "/home/hupeng/code/github/EasyNN/example/res18.pnnx.bin"),0);
    //net.printLayer();
    EXPECT_EQ(net.layer_num,51);
    EXPECT_EQ(net.blob_num,50);
}

TEST(net,extractBlob)
{   

    easynn::Net net;
    EXPECT_EQ(net.loadParam("/home/hupeng/code/github/EasyNN/example/res18.pnnx.param",\
    "/home/hupeng/code/github/EasyNN/example/res18.pnnx.bin"),0);
    net.extractBlob(6);

}