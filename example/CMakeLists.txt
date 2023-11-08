
find_package(OpenCV QUIET COMPONENTS core highgui imgproc)

link_directories(${CMAKE_SOURCE_DIR}/build/src/)

file(GLOB  cpp_files ${EXAMPLE}/*.cpp)
foreach(cpp_file ${cpp_files})
    get_filename_component(FILE_NAME ${cpp_file} NAME)
    string(REPLACE ".cpp" "" FILE_NAME "${FILE_NAME}")   
    add_executable(${FILE_NAME} ${cpp_file})
    target_include_directories(${FILE_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${INCLUDE})
    target_link_libraries(${FILE_NAME} PRIVATE  ${OpenCV_LIBS} easynn)
endforeach()