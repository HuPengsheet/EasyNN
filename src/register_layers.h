#ifndef EASYNN_REGISTER_LAYERS_H
#define EASYNN_REGISTER_LAYERS_H

#include<string>
#include<vector>
#include<utility>

#include"layer.h"
#include"layers/input.h"
#include"layers/output.h"
#include"layers/relu.h"
#include"layers/adaptiveavgpool.h"
#include"layers/convolution.h"
#include"layers/expression.h"
#include"layers/flatten.h"
#include"layers/linear.h"
#include"layers/maxpool.h"
#include"layers/cat.h"
#include"layers/contiguous.h"
#include"layers/permute.h"
#include"layers/silu.h"
#include"layers/upsample.h"
#include"layers/view.h"

#define register_layer(name) \
    easynn::Layer* name##Factory()\
    {\
        return new easynn::name;\
    }


typedef  easynn::Layer*(*layer_factory)();   //定义了一个函数指针，Layer*类型

register_layer(Input);
register_layer(Output);
register_layer(Relu);
register_layer(AdaptivePool);
register_layer(Convolution);
register_layer(Expression);
register_layer(Flatten);
register_layer(Linear);
register_layer(MaxPool);
register_layer(Cat);
register_layer(Contiguous);
register_layer(Permute);
register_layer(Silu);
register_layer(Upsample);
register_layer(View);



std::vector<std::pair<std::string,layer_factory>> layes_factory={
    {"Input",InputFactory},
    {"Output",OutputFactory},
    {"ReLU",ReluFactory},
    {"AdaptiveAvgPool2d",AdaptivePoolFactory},
    {"Conv2d",ConvolutionFactory},
    {"Expression",ExpressionFactory},
    {"flatten",FlattenFactory},
    {"Linear",LinearFactory},
    {"MaxPool2d",MaxPoolFactory},
    {"cat",CatFactory},
    {"contiguous",ContiguousFactory},
    {"permute",PermuteFactory},
    {"silu",SiluFactory},
    {"Upsample",UpsampleFactory},
    {"view",ViewFactory}
    };

#endif