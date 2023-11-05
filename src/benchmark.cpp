
#include <chrono>
#include <sys/time.h> 
#include <pthread.h>  
#include "benchmark.h"
namespace easynn{

double get_current_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
}

}

   
