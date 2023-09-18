cmake_minimum_required(VERSION 3.18)

project(scratchnn)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set_property(GLOBAL PROPERTY LINKER_LANGUAGE CXX)
add_compile_options(-Wall -Werror -Wextra)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(BIN_DIR ${PROJECT_SOURCE_DIR}/bin)

include_directories(${INCLUDE_DIR})

set(SRC src/Tensor.cpp)

add_library(-scratchnn SHARED ${SRC})
set_target_properties(-scratchnn PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(-scratchnn PUBLIC include)

add_subdirectory(tests)
