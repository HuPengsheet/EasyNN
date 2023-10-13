#include"test_ulti.h"
#include"test_mat.h"

TEST(Mat, 1)
{
    easynn::Mat m(10,10);
    //print_mat(m);
}

TEST(Mat, compare_mat)
{   
    
    easynn::Mat m1(10,10);
    easynn::Mat m2(10,10);
    easynn::Mat m3(100,10);
    easynn::Mat m4;
    EXPECT_EQ(compare_mat(m1,m2), 1);
    EXPECT_EQ(compare_mat(m1,m2), 1);
    EXPECT_EQ(compare_mat(m1,m3), 0);
    EXPECT_EQ(compare_mat(m4,m3), 0);
}

TEST(Mat, refcount)
{   
    
    easynn::Mat m1(10,10);
    EXPECT_EQ(*m1.refcount, 1);
    easynn::Mat m2(m1);
    easynn::Mat m3=m1;
    EXPECT_EQ(*m2.refcount, 2);
    EXPECT_EQ(*m3.refcount, 3);

}




int main()
{
    InitQTest();
    return RUN_ALL_TESTS();
}

