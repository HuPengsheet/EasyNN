#include"net.h"
#include<iostream>

namespace easynn{


Net::Net()
{

}

int Net::forwarLayer(std::string layer_name)
{
    
}


int Net::blobforLayer(const size_t blob_num)
{
    pnnx::Operand* blob = graph.operands[blob_num];
    pnnx::Operator* layer = blob->producer;
    if(blob_mat[blob_num].isEmpty())
    {
        pnnx::Operand* blob = graph.operands[blob_num];
        pnnx::Operator* layer = blob->producer;
    }

    
    // if()
    // std::cout<<blob->producer->name<<std::endl;
} 

int Net::extractBlob(const size_t num) 
{
    pnnx::Operand* blob = graph.operands[blob_num];
    int re = -1;
    if(num>blob_num-1 || num<0)
        return -1;
    blob_mat[blob_num].isEmpty();
    pnnx::Operator* layer = blob->producer;
    forwarLayer(layer->name);
}

void Net::printLayer() const
{
    for(auto layer:graph.ops)
    {
        std::cout<<layer->name<<std::endl;
    }
}

int Net::loadParam(const char * param_path,const char * bin_path)
{   
    int re =-1;
    re = graph.load(param_path,bin_path);
    if(re==0)
    {
        layer_num = graph.ops.size();
        blob_num = graph.operands.size();
        blob_mat.resize(blob_num); 
        layers.resize(layer_num);
        for(auto layer:graph.ops)
        {
            layers.push_back(layer->name);
        }
    }
    return re;
}

}