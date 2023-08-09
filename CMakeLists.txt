cmake_minimum_required(VERSION 3.18)

project(scratchnn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set_property(GLOBAL PROPERTY LINKER_LANGUAGE CXX)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(BIN_DIR ${PROJECT_SOURCE_DIR}/bin)

include_directories(${INCLUDE_DIR})

set(SRC include/Tensor.hpp)

add_library(lib-scratchnn SHARED ${SRC})
set_target_properties(lib-scratchnn PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(lib-scratchnn PUBLIC include)

add_subdirectory(tests)
