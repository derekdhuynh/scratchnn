cmake_minimum_required(VERSION 3.18)

project(scratchnn_src)
set(CMAKE_CXX_STANDARD 20)

set(SRC 
    Tensor_test.cpp
    Autograd_test.cpp
)

add_executable(scratchnn_tests ${SRC})
set_target_properties(scratchnn_tests PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(scratchnn_tests PUBLIC ../include)
