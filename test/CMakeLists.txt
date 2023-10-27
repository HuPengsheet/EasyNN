cmake_minimum_required(VERSION 3.10)
project(test)

# set(SRC src/mat.cpp test_mat.cpp)
# set(INCLUDE src/)
set(TEST_SRC  ${TEST}/test.cpp)


add_executable(run_test ${TEST_SRC} ${SRC} ${LAYERS_SRC})
target_include_directories(run_test PUBLIC ${INCLUDE})

