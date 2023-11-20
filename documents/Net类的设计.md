# Net类的设计

```c++
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
```

​	Net类是对pnnx类的解析和重封装，EasyNN采用的是pnnx作为模型的中间结构，有关pnnx的内容大家请参考[pnnx](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)官网的介绍。

## 属性值

```c++
std::vector<Blob> blobs;
std::vector<Mat> blob_mats;
std::vector<Layer* > layers;

size_t layer_num;
size_t blob_num;
Optional op;
```

​	`blobs`记录的是模型文件里所有的blob，`blob_mats`存储的是blob对应的Mat数据，`layers`是模型所有算子的集合，`layer_num`和`blob_num`就是对应的算子个数和blob个数，`op`用来控制算子推理时的一些选项。

## 方法

### loadModel

```c++
int Net::loadModel(const char * param_path,const char * bin_path)
{   
    int re =-1;
    re = graph->load(param_path,bin_path);  //使用PNNX中的方法，先将模型的参数和权重进行加载
    if(re==0)
    {
        layer_num = graph->ops.size();
        blob_num = graph->operands.size();
        blobs.resize(blob_num); 
        blob_mats.resize(blob_num); 
        layers.resize(layer_num);

        //遍历算子集合
        for(int i=0;i<layer_num;i++)
        {
            pnnx::Operator* op = graph->ops[i]; 
            std::string layer_type = extractLayer(op->type);   //提取算子的名字
            layer_factory factory = 0;
            for(auto l:layes_factory)
            {
                if(layer_type==l.first) factory=l.second;   //根据算子的名字，查找出对应的算子工厂
            }
            if(!factory)
            {
                printf("%s is not supportl\n",layer_type.c_str());   //如果没有这个算子，则报错退出
                re=-1;
                break;
            }
            Layer* layer = factory();   //使用算子工厂，实例化算子
            
            layer->name = op->name;     //初始化算子名字
            layer->type = layer_type;	//初始化算子类型

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

            layer->loadParam(op->params);  //加载算子的参数
            layer->loadBin(op->attrs);	   //加载算子的权重
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
```

​	总体思路就是，把先把模型的参数和权重加载到pnnx::graph里面去，然后对pnnx::graph提取出我们自己想要的信息，去初始化EasyNN里面，我们用的到的信息。

### printLayer

```c++
void Net::printLayer() const
{
    for(auto layer:graph->ops)
    {
        printf("%s \n",layer->name.c_str());
    }
}
```

​	遍历所有的算子，并打印算子的名字。

### input

```
int Net::input(int index,const Mat& input)
{
    blob_mats[index]=input;
    return 0;
}
```

​	把数据放到指定位置的blob_mats中，一般情况下就是用来放整个模型的输出的

### extractBlob

```c++
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
```

​	提取出模型某个blob的数据，一般也就是我们要的整个模型的输出。如果它为空，则表示产生它的算子没有被执行，因此需要调用forwarLayer执行这个算子，才能获得我们想要的数据。

### forwarLayer

```c++

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
```

​	`forwarLayer`是一个递归函数，我们要执行一个算子，那这个算子的输入blob_mat必须要存在，如果不存在则需要递归去执行产生这个blob的算子。当某一个算子的输入blob已经存在时，也就是我们手工用input放置在blob_mats中的数据，此时达到递归终止条件，开始一个一个执行算子。根据算子的one_blob_only参数，分别调用不同的`layer->forward`函数。

