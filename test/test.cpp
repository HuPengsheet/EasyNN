#include"test_ulti.h"
#include"test_mat.h"
#include"test_layer.h"
#include"test_conv.h"
#include"test_relu.h"
#include"test_maxpool2d.h"
#include"test_expression.h"
#include"test_adaptiveavgpool.h"
#include"test_flatten.h"
#include"test_linear.h"
#include"test_silu.h"
#include"test_upsample.h"
#include"test_view.h"


int main()
{
    InitQTest();
    return RUN_ALL_TESTS();
}

