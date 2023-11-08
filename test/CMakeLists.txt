link_directories(${CMAKE_SOURCE_DIR}/build/src/)

set(TEST_SRC  ${TEST}/test.cpp)

add_executable(run_test ${TEST_SRC})
target_include_directories(run_test PUBLIC ${INCLUDE})
target_link_libraries(run_test PRIVATE  easynn)

