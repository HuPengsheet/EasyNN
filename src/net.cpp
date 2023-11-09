#include"net.h"
#include"register_layers.h"



namespace easynn{

//提取layers的类型 如输入nn.Conv2d 输出Conv2d
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
    graph = new pnnx::Graph;
}

Net::~Net()
{
    clear();
    for(auto layer:layers)
        delete layer;
}

int Net::clear()
{
    for(auto &m:blob_mats)
    {
        m.clean();
    }
    return 0;
}

int Net::forwarLayer(int layer_index)
{   

    if(layer_index>layer_num-1 || layer_index<0)
    {
        printf("do not have this layer ,layer num is %d",layer_index);
        return -1;
    }
        
    Layer* layer = layers[layer_index];
    for(auto input:layer->bottoms)
    {
        if(blob_mats[input].isEmpty())
            forwarLayer(blobs[input].producer);   //递归调用，直到找到某个layer的输入blob已经存在，说明此时可以forwarLayer
    }
    if(layer->one_blob_only)
    {
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];
        int re = layer->forward(blob_mats[bottom_blob_index],blob_mats[top_blob_index],op);
        if(re!=0)
        {
            printf("%s forward fail",layer->name.c_str());
            return -1;
        }
    }
    else
    {
        std::vector<Mat> input_mats(layer->bottoms.size());
        std::vector<Mat> output_mats(layer->tops.size());

        for(int i=0;i<layer->bottoms.size();i++)
        {
            input_mats[i] = blob_mats[layer->bottoms[i]];
        }

        int re = layer->forward(input_mats,output_mats,op);

        if(re!=0)
        {
            printf("%s forward fail",layer->name.c_str());
            return -1;
        }

        for(int i=0;i<layer->tops.size();i++)
        {
            blob_mats[layer->tops[i]]=output_mats[i];
        }       

    }
    return 0;
}

int Net::input(int index,const Mat& input)
{
    blob_mats[index]=input;
    return 0;
}

int Net::extractBlob(const size_t num,Mat& output) 
{
    Blob& blob = blobs[num];
    if(num>blob_num-1 || num<0)
    {
        printf("the %ld blob is not exist ,please check out\n",num);
        return -1;
    }
        
    if(blob_mats[num].isEmpty())
        forwarLayer(blob.producer);
    
    output = blob_mats[num];
    return 0;
}

void Net::printLayer() const
{
    for(auto layer:graph->ops)
    {
        printf("%s \n",layer->name.c_str());;
    }
}

int Net::loadModel(const char * param_path,const char * bin_path)
{   
    int re =-1;
    re = graph->load(param_path,bin_path);
    if(re==0)
    {
        layer_num = graph->ops.size();
        blob_num = graph->operands.size();
        blobs.resize(blob_num); 
        blob_mats.resize(blob_num); 
        layers.resize(layer_num);

        for(int i=0;i<layer_num;i++)
        {
            pnnx::Operator* op = graph->ops[i]; 
            std::string layer_type = extractLayer(op->type);
            layer_factory factory = 0;
            for(auto l:layes_factory)
            {
                if(layer_type==l.first) factory=l.second;   //根据算子的名字，查找出对应的算子工厂
            }
            if(!factory)
            {
                printf("%s is not supportl\n",layer_type.c_str());
                re=-1;
                break;
            }
            Layer* layer = factory();   //使用算子工厂，实例化算子
            
            layer->name = op->name;
            layer->type = layer_type;

            //构建计算关系，每个layer的输入输出blob是哪个，每个blob是哪个layer产生，是哪个layer使用
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
            layer->loadBin(op->attrs);
            layers[i]= layer;
        }
        delete graph;    //加载完成后，释放PNNX中的图
    }
    else
    {
        printf("load %s  %s fail\n",param_path,bin_path);
        return re;
    }
    return re;
}

}