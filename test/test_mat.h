#include<stdio.h>
#include"test_ulti.h"
#include"mat.h"

void print_mat(const easynn::Mat& m)
{
    if(m.is_empty())
    {
        printf("mat is empty");
        return ;
    }  
    for(int i=0;i<m.c;i++){
        float *ptr = (float *)((char *)m.data+m.cstep*i);
        for(int j=0;j<m.h;j++){
            for(int k=0;k<m.w;k++){
                printf("%f ",ptr[k]);
            }
            ptr = ptr+m.w;
        }
    }
        
}


TEST(c, 1)
{
    easynn::Mat m(10,10);
    print_mat(m);
}