#include<iostream>
#include"net.h"
#include"register_layers.h"



namespace easynn{

std::string extractLayer(const std::string& input) {
    std::string afterDot;
    size_t dotPos = input.find(".");
    if (dotPos != std::string::npos) {
        afterDot = input.substr(dotPos + 1);
    }
    return afterDot;
}

Net::Net()
{
    op = Optional();
}

int Net::forwarLayer(int layer_index)
{   

    if(layer_index>layer_num-1 || layer_index<0)
        return -1;
    Layer* layer = layers[layer_index];
    for(auto input:layer->bottoms)
    {
        if(blob_mats[input].isEmpty())
            forwarLayer(blobs[input].producer);
    }
    if(layer->one_blob_only)
    {
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];
        layer->forward(blob_mats[bottom_blob_index],blob_mats[top_blob_index],op);
    }
    else
    {
        std::vector<Mat> input_mats(layer->bottoms.size());
        std::vector<Mat> output_mats(layer->tops.size());

        for(int i=0;i<layer->bottoms.size();i++)
        {
            input_mats[i] = blob_mats[layer->bottoms[i]];
        }

        layer->forward(input_mats,output_mats,op);

        for(int i=0;i<layer->tops.size();i++)
        {
            blob_mats[layer->tops[i]]=output_mats[i];
        }       

    }
    return 0;
}


int Net::blobforLayer(const size_t blob_num)
{
    pnnx::Operand* blob = graph.operands[blob_num];
    pnnx::Operator* layer = blob->producer;
    if(blob_mats[blob_num].isEmpty())
    {
        pnnx::Operand* blob = graph.operands[blob_num];
        pnnx::Operator* layer = blob->producer;
    }

    return 0;
    // if()
    // std::cout<<blob->producer->name<<std::endl;
} 

int Net::extractBlob(const size_t num,Mat& output) 
{
    Blob& blob = blobs[num];
    int re = -1;
    if(num>blob_num-1 || num<0)
        return re;
    if(blob_mats[num].isEmpty())
        forwarLayer(blob.producer);
    output = blob_mats[num];
    return 0;
}

void Net::printLayer() const
{
    for(auto layer:graph.ops)
    {
        std::cout<<layer->name<<std::endl;
    }
}

int Net::loadModel(const char * param_path,const char * bin_path)
{   
    int re =-1;
    re = graph.load(param_path,bin_path);
    if(re==0)
    {
        layer_num = graph.ops.size();
        blob_num = graph.operands.size();
        blobs.resize(blob_num); 
        blob_mats.resize(blob_num); 
        layers.resize(layer_num);
        for(int i=0;i<layer_num;i++)
        {
            pnnx::Operator* op = graph.ops[i]; 
            std::string layer_type = extractLayer(op->type);
            layer_factory factory = 0;
            for(auto l:layes_factory)
            {
                if(layer_type==l.first) factory=l.second;
            }
            if(!factory)
            {
                std::cout<<layer_type<<" is not support"<<std::endl;
                re=-1;
                break;
            }
            Layer* layer = factory();
            
            layer->name = op->name;
            layer->type = layer_type;
            for(auto input:op->inputs)
            {
                int blob_index = std::stoi(input->name);
                layer->bottoms.push_back(blob_index);
                Blob& blob = blobs[blob_index];
                blob.consumer = i;
            }
            for(auto output:op->outputs)
            {
                int blob_index = std::stoi(output->name);
                layer->tops.push_back(blob_index);
                Blob& blob = blobs[blob_index];
                blob.producer = i;
            }
            layer->loadParam(op->params);

            layers[i]= layer;
        }
    }

    return re;
}

}