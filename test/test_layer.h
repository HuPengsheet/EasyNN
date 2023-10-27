#ifndef EASYNN_TETS_LAYER_H
#define EASYNN_TETS_LAYER_H
#include"test_ulti.h"
#include"layers/convolution.h"
#include"layers/relu.h"
#include"layers/adaptiveavgpool.h"
#include"layers/expression.h"
#include"layers/flatten.h"
#include"layers/linear.h"
#include"layers/maxpool.h"

TEST(layer,forward)
{
    
    easynn::Convolution c1;
    c1.forward();

    easynn::Relu re;
    re.forward();

    easynn::AdaptivePool ad;
    ad.forward();

    easynn::Flatten fla;
    fla.forward();

    easynn::Linear linear;
    linear.forward();

    easynn::MaxPool mp;
    mp.forward();

    easynn::Expression ex;
    ex.forward();


}


#endif