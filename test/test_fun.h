#ifndef EASYNN_TETS_FUN_H
#define EASYNN_TETS_FUN_H

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
    for (int q=0; q<m.c; q++)
    {
        float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}


// int compareMat(const easynn::Mat& m1,const easynn::Mat& m2)
// {
//     if(m1.isEmpty()||m2.isEmpty())
//     {
//         //printf("mat is empty,con't compare\n");
//         return -1;
//     } 
//     if(m1.c!=m2.c || m1.h!=m2.h || m1.w!=m2.w || m1.total()!=m2.total())
//     {
//         //printf("mat is shape is different,con't compare\n");
//         return -1;
//     }
//     return !memcmp(m1.data, m2.data,m1.total());
// }

int compareMat(const easynn::Mat& m1,const easynn::Mat& m2)
{
    if(m1.isEmpty()||m2.isEmpty())
    {
        //printf("mat is empty,con't compare\n");
        return -1;
    } 
    if(m1.c!=m2.c || m1.h!=m2.h || m1.w!=m2.w || m1.total()!=m2.total())
    {
        //printf("mat is shape is different,con't compare\n");
        return -1;
    }
    for (int q=0; q<m1.c; q++)
    {
        float* ptr1 = m1.channel(q);
        float* ptr2 = m2.channel(q);
        for (int z=0; z<m1.d; z++)
        {
            for (int y=0; y<m1.h; y++)
            {
                for (int x=0; x<m1.w; x++)
                {
                    if(abs(ptr1[x]-ptr2[x])>1e-2)
                        return -1;
                }
                ptr1 += m1.w;
                ptr2 += m2.w;
            }
        }
    }
    return 0;
}

#endif