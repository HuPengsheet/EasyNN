#ifndef EASYNN_NET_H
#define EASYNN_NET_H

#include"ir.h"
namespace easynn {
    

class Net
{
public:
    Net();
    virtual ~Net();

private:
    pnnx::Graph graph;
};
                                  

}




#endif