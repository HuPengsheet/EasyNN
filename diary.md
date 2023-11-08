##10月8日
    折腾了好几天打算先写起来吧，不管怎么样，边写边改。
    在allocator.h里下了fastMolloc和fastFree来分配和释放内存，mat类开了个头
    在想写一个简单的测试框架用来测试，一个头文件那种就能搞定的，在网上找了一下还真找到一个合适的
    把它放在了test/test_ulti.h下，具体的代码明天再看了，今天太晚了。


##10月9日       
    今天在学校装了一天的小车架子，根本没时间看代码，外卖都放校门口一个多小时已经凉了。
    test/test_ulti.h的代码后面再看吧，先放在这了，也不会拖太久，写完mat类肯定就要做对应的测试。
    说实话我连mat类的构造函数写的不利索，什么拷贝构造不是很清楚，运算符重载也没写过。
    边写边补吧，不过心里有个大概得框架就还行。

##10月10日     
    今天大概写了一小时mat类，发现mat的whc可能设置为int比较好
    因为后面有各种函数重载，这样吧int和size_t用来区分函数重载
    我说ncnn为什么用int类型

##10月12日    
    Mat写了个大概，正在写Mat类的测试代码，用那个简单的测试框架
    把CMakeLists重写了一下，分多个cmake
    这要可以在主目录下的CMakeLists.txt里传变量的值，编译起来就比较方便了


##10月14日     
    继续写Mat类，写好了构造函数,引用计数，赋值。对应的测试也块完善
    改了一下函数的名字，尽量统一命名规范。

##10月15日     
    Mat类基本已经写好了，各种构造函数，引用计数，内存管理与释放，这些都写好了
    对应的单元测试也都写了，其他的什么reshape，归一化这些后面再写吧
    


##10月26日      
    导师安排去出差了，10来天没更新了。昨天刚回来。
    这段时间构思了一下架构，准备先把layer写了，然后再写net类。


==109145== 
==109145== HEAP SUMMARY:
==109145==     in use at exit: 1,120,274 bytes in 8,585 blocks
==109145==   total heap usage: 58,955 allocs, 50,370 frees, 414,600,313 bytes allocated
==109145== 
==109145== LEAK SUMMARY:
==109145==    definitely lost: 0 bytes in 0 blocks
==109145==    indirectly lost: 0 bytes in 0 blocks
==109145==      possibly lost: 7,472 bytes in 39 blocks
==109145==    still reachable: 1,039,714 bytes in 8,096 blocks
==109145==                       of which reachable via heuristic:
==109145==                         length64           : 5,408 bytes in 77 blocks
==109145==                         newarray           : 1,760 bytes in 30 blocks
==109145==         suppressed: 0 bytes in 0 blocks
==109145== Rerun with --leak-check=full to see details of leaked memory
==109145== 
==109145== For lists of detected and suppressed errors, rerun with: -s
==109145== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)



需要写的算子：
   Input
   Output

   Conv2d
   Linear

   MaxPool2d 
   Upsample
   AdaptiveAvgPool2d
   
   flatten 
   view
   permute
   Expression
   cat

   silu
   ReLU