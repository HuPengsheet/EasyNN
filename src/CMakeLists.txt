
find_package(OpenMP)
add_library(easynn STATIC ${SRC} ${LAYERS_SRC} ${CUDA_SRC})
target_include_directories(easynn PUBLIC ${INCLUDE} )
if(OpenMP_CXX_FOUND)
    target_link_libraries(easynn OpenMP::OpenMP_CXX)
endif()