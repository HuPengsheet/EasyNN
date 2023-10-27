#ifndef EASYNN_REGISTER_LAYERS_H
#define EASYNN_REGISTER_LAYERS_H

#include<string>
#include<vector>
#include<utility>

#include"layer.h"
#include"layers/relu.h"

#define register_layer(name) \
    easynn::Layer* name##Factory()\
    {\
        return new easynn::name;\
    }

typedef  easynn::Layer*(*layer_factory)();

register_layer(Relu);

std::vector<std::pair<std::string,layer_factory>> layes_factory={
    {"ReLU",ReluFactory}
    };

#endif