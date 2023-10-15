#include"test_ulti.h"
#include"test_fun.h"
#include<vector>


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
    easynn::Mat m2(3,5);
    easynn::Mat m3(2,3,2);
    std::vector<int> x1={1,2,3,4,5};
    std::vector<std::vector<int>> x2={{1,2,2},{2,98,100},{3,4,5},{31,4,52},{3,43,59}};
    std::vector<std::vector<std::vector<int>>> x3={{{1,2},{98,100},{4,5}},{{5,2},{99,100},{4,51}}};
    m1.fillFromArray(x1);
    m2.fillFromArray(x2);
    m3.fillFromArray(x3);
    // printMat(m1);
    // printMat(m2);
    // printMat(m3);
}
 
TEST(Mat,clone)
{   
    easynn::Mat m1(5);
    easynn::Mat m2(3,5);
    easynn::Mat m3(2,3,2);
    std::vector<int> x1={1,2,3,4,5};
    std::vector<std::vector<int>> x2={{1,2,2},{2,98,100},{3,4,5},{31,4,52},{3,43,59}};
    std::vector<std::vector<std::vector<int>>> x3={{{1,2},{98,100},{4,5}},{{5,2},{99,100},{4,51}}};
    m1.fillFromArray(x1);
    m2.fillFromArray(x2);
    m3.fillFromArray(x3);
    
    easynn::Mat m4 = m1.clone();
    easynn::Mat m5 = m2.clone();
    easynn::Mat m6 = m3.clone();
    // printMat(m1);
    //printMat(m4);
    EXPECT_EQ(compareMat(m1,m4), 1);
    EXPECT_EQ(compareMat(m2,m5), 1);
    EXPECT_EQ(compareMat(m3,m6), 1);

}