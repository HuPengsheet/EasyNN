#ifndef EASYNN_REGISTER_LAYERS_H
#define EASYNN_REGISTER_LAYERS_H

#include<string>
#include<vector>
#include<utility>

#include"layer.h"
#include"layers/cxx/input.h"
#include"layers/cxx/output.h"
#include"layers/cxx/relu.h"
#include"layers/cxx/adaptiveavgpool.h"
#include"layers/cxx/convolution.h"
#include"layers/cxx/expression.h"
#include"layers/cxx/flatten.h"
#include"layers/cxx/linear.h"
#include"layers/cxx/maxpool.h"
#include"layers/cxx/cat.h"
#include"layers/cxx/contiguous.h"
#include"layers/cxx/permute.h"
#include"layers/cxx/silu.h"
#include"layers/cxx/upsample.h"
#include"layers/cxx/view.h"

#define register_layer(name) \
    easynn::Layer* name##Factory()\
    {\
        return new easynn::name;\
    }


typedef  easynn::Layer*(*layer_factory)();   

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