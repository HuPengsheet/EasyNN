
#include"test_ulti.h"
#include"test_fun.h"
#include"ir.h"
#include"net.h"
#include"mat.h"
TEST(net,printLayer)
{
    easynn::Net net;
    EXPECT_EQ(net.loadModel("/home/hupeng/code/github/EasyNN/example/res18.pnnx.param",\
    "/home/hupeng/code/github/EasyNN/example/res18.pnnx.bin"),0);

    EXPECT_EQ(net.layer_num,51);
    EXPECT_EQ(net.blob_num,50);
    easynn::Mat m1(10,10);
    m1.fill(5);
    net.blob_mats[0]=m1;
    easynn::Mat m2;
    net.extractBlob(49,m2);
    printMat(m2);
}

TEST(net,yolov5)
{
    // easynn::Net net;
    // EXPECT_EQ(net.loadModel("/home/hupeng/code/github/EasyNN/example/yolov5s.torchscript.pnnx.param",\
    // "/home/hupeng/code/github/EasyNN/example/yolov5s.torchscript.pnnx.bin"),0);

    // easynn::Mat m1(10,10);
    // m1.fill(5);
    // net.blob_mats[0]=m1;
    // easynn::Mat m2;
    // net.extractBlob(152,m2);
    // printMat(m2);

}
TEST(net,extractBlob)
{   

    // easynn::Net net;
    // EXPECT_EQ(net.loadParam("/home/hupeng/code/github/EasyNN/example/res18.pnnx.param",\
    // "/home/hupeng/code/github/EasyNN/example/res18.pnnx.bin"),0);
    // // net.extractBlob(6);

}