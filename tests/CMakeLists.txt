cmake_minimum_required(VERSION 3.18)

project(scratchnn_test)
set(CMAKE_CXX_STANDARD 23)

set(SRC test_main.cpp)

add_executable(scratchnn_tests ${SRC})
set_target_properties(scratchnn_tests PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(scratchnn_tests PUBLIC ../include)
target_link_libraries(scratchnn_tests -scratchnn)
