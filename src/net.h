#ifndef EASYNN_NET_H
#define EASYNN_NET_H

#include"ir.h"
#include"mat.h"
#include<vector>
namespace easynn {
    

class Net
{
public:
    Net();
    // ~Net();
    void printLayer() const;
    int loadParam(const char * param_path,const char * bin_path);
    int extractBlob(const size_t num);
    int blobforLayer(const size_t blob_num);
    int forwarLayer(std::string layer_name);


    std::vector<easynn::Mat> blob_mat;
    std::vector<std::string> layers;
    
    size_t layer_num;
    size_t blob_num;

private:
    pnnx::Graph graph;
};
                                  

}




#endif