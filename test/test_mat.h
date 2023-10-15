#include<stdio.h>
#include<stdlib.h>
#include"mat.h"

//Print Mat
void printMat(const easynn::Mat& m)
{
    if(m.isEmpty())
    {
        printf("mat is empty\n");
        return ;
    } 
    printf("d=%d,c=%d,h=%d,w=%d \n",m.d,m.c,m.h,m.w);
    for(int i=0;i<m.c;i++){
        float *ptr = (float *)((char *)m.data+m.cstep*i);
        for(int d=0;d<m.d;d++){
            for(int j=0;j<m.h;j++){
                for(int k=0;k<m.w;k++){
                    printf("%f ",ptr[k]);
                }
                printf("\n");
                ptr = ptr+m.w;
            }
            printf("\n");
        }
        printf("\n");
    }
}


int compareMat(const easynn::Mat& m1,const easynn::Mat& m2)
{
    if(m1.isEmpty()||m2.isEmpty())
    {
        //printf("mat is empty,con't compare\n");
        return 0;
    } 
    if(m1.c!=m2.c || m1.h!=m2.h || m1.w!=m2.w || m1.total()!=m2.total())
    {
        //printf("mat is shape is different,con't compare\n");
        return 0;
    }
    return !memcmp(m1.data, m2.data,m1.total());
}
