#include"test_ulti.h"
#include"test_mat.h"

TEST(Mat, print_mat)
{
    easynn::Mat m1;
    easynn::Mat m2(2,3);
    easynn::Mat m3(2,3,2);
    easynn::Mat m4(2,3,2,3);
    //printMat(m1);
    //printMat(m2);
    //printMat(m3);
    //printMat(m4);
}

TEST(Mat, compare_mat)
{   
    easynn::Mat m1(10,10);
    easynn::Mat m2(10,10);
    easynn::Mat m3(100,10);
    easynn::Mat m4;
    EXPECT_EQ(compareMat(m1,m2), 1);
    EXPECT_EQ(compareMat(m1,m2), 1);
    EXPECT_EQ(compareMat(m1,m3), 0);
    EXPECT_EQ(compareMat(m4,m3), 0);
}

TEST(Mat, refcount)
{   
    {
        easynn::Mat m1(10,10);
        EXPECT_EQ(*m1.refcount, 1);
        easynn::Mat m2(m1);
        easynn::Mat m3=m1;
        EXPECT_EQ(*m2.refcount, 3);
        EXPECT_EQ(*m3.refcount, 3);
    }

    easynn::Mat m1(10,10);
    {   
        EXPECT_EQ(*m1.refcount, 1);
        easynn::Mat m2(m1);
        easynn::Mat m3=m1;
        EXPECT_EQ(*m2.refcount, 3);
        EXPECT_EQ(*m3.refcount, 3);
    }
    EXPECT_EQ(*m1.refcount, 1);

    {
        
        easynn::Mat m2;
        easynn::Mat m3=m2;
        EXPECT_EQ((long)m2.refcount, 0);
        EXPECT_EQ((long)m3.refcount, 0);
    }

}

TEST(Mat, fill)
{   
    easynn::Mat m1;
    easynn::Mat m2(3);
    easynn::Mat m3(3,4);
    easynn::Mat m4(3,4,2);
    easynn::Mat m5(3,4,2,2);
    m1.fill(2);
    m2.fill(2.2f);
    m3.fill(2);
    m4.fill(2);
    m5.fill(2.2f);
    //printMat(m3);
}
TEST(Mat,fillFromArray)
{   
    easynn::Mat m1(5);
    int array[]={0,1,2,3,4};
    m1.fillFromArray(array);
    printf("***%d\n",sizeof(array));
    printMat(m1);
}

int main()
{
    InitQTest();
    return RUN_ALL_TESTS();
}

