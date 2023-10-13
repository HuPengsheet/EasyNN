#include<stdio.h>
#include<stdlib.h>
#include"mat.h"

//打印Mat
void print_mat(const easynn::Mat& m)
{
    if(m.is_empty())
    {
        printf("mat is empty");
        return ;
    } 
    printf("channel=%d,h=%d,w=%d \n",m.c,m.h,m.w);
    for(int i=0;i<m.c;i++){
        float *ptr = (float *)((char *)m.data+m.cstep*i);
        for(int j=0;j<m.h;j++){
            for(int k=0;k<m.w;k++){
                printf("%f ",ptr[k]);
            }
            printf("\n");
            ptr = ptr+m.w;
        }
        printf("\n");
    }
    printf("channel=%d,h=%d,w=%d \n",m.c,m.h,m.w);  
}

//相同返回1，不同返回0
int compare_mat(const easynn::Mat& m1,const easynn::Mat& m2)
{
    if(m1.is_empty()||m2.is_empty())
    {
        printf("mat is empty,con't compare\n");
        return 0;
    } 
    if(m1.c!=m2.c || m1.h!=m2.h || m1.w!=m2.w || m1.total()!=m2.total())
    {
        printf("mat is shape is different,con't compare\n");
        return 0;
    }
    return !memcmp(m1.data, m2.data,m1.total());
}
