#ifndef EASYNN_NET_H
#define EASYNN_NET_H

#include<vector>
#include"ir.h"
#include"mat.h"
#include"blob.h"
#include"layer.h"
#include"optional.h"
namespace easynn {
    

class Net
{
public:
    Net();
    ~Net();
    void printLayer() const;
    int loadModel(const char * param_path,const char * bin_path);
    int extractBlob(const size_t num,Mat& output);
    int forwarLayer(int layer_index);
    int input(int index,const Mat& input);
    int clear();

    std::vector<Blob> blobs;
    std::vector<Mat> blob_mats;
    std::vector<Layer* > layers;

    size_t layer_num;
    size_t blob_num;
    Optional op;
private:
    pnnx::Graph* graph;
};
                                  

}




#endif