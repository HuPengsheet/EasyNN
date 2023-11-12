<img src="./images/logo.png" />

​	EasyNN是一个面向教学而研发的推理框架，旨在帮助大家在最短时间内写出一个支持ResNet和YOLOv5等模型的深度学习推理框架。**简单**是EasyNN最大的特点，只需要掌握C++基本语法和神经网络基础知识，你就可以在15天内，写出一个属于自己的网络推理框架。

# 特性

- **无第三方库依赖**：EasyNN内含简单的测试框架，并实现了一个简单的Mat数据类。
- **使用PNNX作为模型中间结构**：EasyNN采用[PNNX](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)作为模型的中间结构，大大提高了开发效率。
- **支持动态尺寸输入**：EasyNN具备支持图片动态尺寸输入的能力。使用递归的方式在运行时构建计算顺序，无需指定固定尺寸。
- **OpenMP多线程加速**：EasyNN利用OpenMP技术实现了多线程加速，以提高推理速度。
- **简洁易读的代码**：EasyNN的代码仅采用部分C++11特性编写，代码简洁易读。
- **完善的开发文档和教程**：在开发文档中详细介绍了每个类和函数的作用，并在B站配备视频讲解，带各位敲每一行代码。

# 你可以学到的内容

- **C++语法和相关概念**：可以熟悉C++的基本语法和常用特性，如类的定义、成员函数的实现、继承、虚函数的使用等。

- **设计模式和编程范式**：EasyNN的开发过程中涉及到一些常见的设计模式，例如工厂模式和引用计数法。

- **框架开发全流程**：通过学习EasyNN的开发，可以了解一个推理框架的完整开发流程。从框架设计、代码实现到单元测试和调试。

- **常见算子的实现方法**：常见的卷积算子(Conv2d)、池化算子(MaxPool2d)等的实现。

    

    # 编译与运行
