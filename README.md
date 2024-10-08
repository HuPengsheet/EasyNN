<img src="./images/logo.png" />

EasyNN是一个面向教学而研发的推理框架，旨在帮助大家在最短时间内写出一个支持ResNet和YOLOv5等模型的深度学习推理框架。**简单**是EasyNN最大的特点，只需要掌握C++基本语法和神经网络基础知识，你就可以在15天内，写出一个属于自己的网络推理框架。

# 特性

- **无第三方库依赖**：EasyNN内含简单的测试框架，并实现了一个简单的Mat数据类。
- **使用PNNX作为模型中间结构**：EasyNN采用[PNNX](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)作为模型的中间结构，大大提高了开发效率。
- **OpenMP多线程加速**：EasyNN利用OpenMP技术实现了多线程加速，以提高推理速度。
- **CUDA加速**：EasyNN部分算子支持CUDA加速,卷积算子支持im2col+CUDA GEMM加速
- **简洁易读的代码**：EasyNN的代码仅采用部分C++11特性编写，代码简洁易读。
- **完善的开发文档和教程**：在开发文档中详细介绍了每个类和函数的作用，并在B站配备视频讲解，带各位敲每一行代码。

# 你可以学到的内容

- **C++语法和相关概念**：可以熟悉C++的基本语法和常用特性，如类的定义、成员函数的实现、继承、虚函数的使用等。
- **设计模式和编程范式**：EasyNN的开发过程中涉及到一些常见的设计模式，例如工厂模式和引用计数法。
- **框架开发全流程**：通过学习EasyNN的开发，可以了解一个推理框架的完整开发流程。从框架设计、代码实现到单元测试和调试。
- **常见算子的实现方法**：常见的卷积算子(Conv2d)、池化算子(MaxPool2d)等的实现。
- **CUDA高性能编程**：CUDA编程模型、GEMM、向量化读取、共享内存.

# 支持的PNNX算子
```c++
register_layer(Input);              //no need
register_layer(Output);             //no need
register_layer(Contiguous);         //no need

register_layer(Relu);               //cuda accelerate
register_layer(Silu);               //cuda accelerate

register_layer(Convolution);        //im2col+sgemm cuda accelerate
register_layer(AdaptivePool);
register_layer(MaxPool);
register_layer(Upsample);

register_layer(Linear);             //cuda accelerate
register_layer(Expression);
register_layer(Flatten);
register_layer(View);
register_layer(Cat);
register_layer(Permute);
```
   

# 编译与运行

第一步：下载并编译代码

```shell
git clone https://github.com/HuPengsheet/EasyNN.git
cd EasyNN
mkdir build && cd build
cmake ..
make -j4
```

第二步：下载对应的模型权重

```shell
方法一：使用wget从github上下载
cd ../example
wget https://github.com/HuPengsheet/EasyNN/releases/download/EasyNN1.0-model-file/model.tar.xz
tar -xf model.tar.xz

方法二：通过百度云下载，把下载好的文件解压到，项目目录下的example下
链接: https://pan.baidu.com/s/1RgbSGVNSfYZZtos6Y4Bedw 提取码: h9u6 
```

第三步：运行res18和yolov5推理代码，可以看到对应的推理结果

```shell
#进入到build目前下

#运行res18的代码
./example/res18

#运行yolov5s的代码
./example/yolov5s
```

第四部（可选）：运行单元测试

```shell
./test/run_test
```

# 开发文档
- [测试框架](https://github.com/HuPengsheet/EasyNN/blob/main/documents/%E6%B5%8B%E8%AF%95%E6%A1%86%E6%9E%B6.md)
- [内存分配](https://github.com/HuPengsheet/EasyNN/blob/main/documents/%E5%86%85%E5%AD%98%E5%88%86%E9%85%8D.md)
- [Mat类设计](https://github.com/HuPengsheet/EasyNN/blob/main/documents/Mat%E7%B1%BB%E7%9A%84%E8%AE%BE%E8%AE%A1.md)
- [Layer类和Blob的设计](https://github.com/HuPengsheet/EasyNN/blob/main/documents/Layer%E7%B1%BB%E5%92%8CBlob%E7%9A%84%E8%AE%BE%E8%AE%A1.md)
- [optional类与benckmark](https://github.com/HuPengsheet/EasyNN/blob/main/documents/optional%E7%B1%BB%E4%B8%8Ebenckmark.md)
- [Net类设计](https://github.com/HuPengsheet/EasyNN/blob/main/documents/Net%E7%B1%BB%E7%9A%84%E8%AE%BE%E8%AE%A1.md)
- 未完待续

# 致谢

本项目中很大一部分代码参考了优秀的推理框架[ncnn](https://github.com/Tencent/ncnn)

其中example下的部分代码借鉴了[KuiperInfer](https://github.com/zjhellofss/KuiperInfer)
