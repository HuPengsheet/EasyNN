cmake_minimum_required(VERSION 3.10)
project(EasyNN)


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)  # 将 C++ 标准设置为 C++ 11
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # C++ 11 是强制要求，不会衰退至低版本
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁止使用编译器特有扩展

add_definitions(-DMY_GLOBAL_MACRO=1)

option(EASTNN_USE_CUDA "enable cuda " OFF)

file(GLOB SRC src/*.cpp)
file(GLOB LAYERS_SRC src/layers/*.cpp)

set(INCLUDE ${CMAKE_SOURCE_DIR}/src)
set(TEST ${CMAKE_SOURCE_DIR}/test)
set(EXAMPLE ${CMAKE_SOURCE_DIR}/example)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(example)