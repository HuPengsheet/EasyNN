# Layer类和Blob的设计

# Layer类

```c++
class Layer{

public:
    Layer();
    virtual ~Layer();
    virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);
    virtual int loadBin(std::map<std::string, pnnx::Attribute>& attrs);
    virtual int forward(const Mat& input,Mat& output,const Optional& op);
    virtual int forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op);  

public:

    bool one_blob_only;  

public:

    std::string type;  
    std::string name;   

    std::vector<int> bottoms;   
    std::vector<int> tops;      

};
```

​	Layer类表示的是一个算子，Layer是一个虚类，当我们要具体实现某一个算子的时候，我们只要继承自Layer，然后重写相关的函数即可。

### 属性值

```c++
public:

    bool one_blob_only;   //这个算子是否单输入输出

public:

    std::string type;   //算子的类型
    std::string name;   //算子的名字

    std::vector<int> bottoms;    //算子的输入blob序号vector
    std::vector<int> tops;       //算子的输出blob序号vector
```

### 方法

```c++
Layer();
virtual ~Layer();
virtual int loadParam(std::map<std::string, pnnx::Parameter>& params);  //加载模型的参数
virtual int loadBin(std::map<std::string, pnnx::Attribute>& attrs);     //加载模型的权重
virtual int forward(const Mat& input,Mat& output,const Optional& op);	//模型推理，对应one_blob_only=true
virtual int forward(const std::vector<Mat>& input,std::vector<Mat>& output,const Optional& op);  //模型推理，对应one_blob_only=flase
```

## Blob类

```c++
class  Blob
{
public:
    // empty
    Blob();

public:

    int producer;
    int consumer;
    Mat shape;
};
```

​	Blob类，记录的是算子的输入输出，producer表示生产该blob算子的序号，consumer使用该blob算子的序号，shape表示尺寸。

