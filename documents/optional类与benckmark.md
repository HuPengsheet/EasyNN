## Optional类

```c++
class Optional
{
public:
    Optional();
    int num_thread;
};
```

​	只有一个参数，`num_thread`表示openmp的多线程个数，默认值为8。

## benckmark

```c++
double get_current_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
}
```

​	返回当前的时间，用于计时。