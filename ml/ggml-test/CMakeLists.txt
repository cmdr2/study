cmake_minimum_required(VERSION 3.12)
project(ggml-test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set(GGML_CUDA ON)

add_subdirectory(ggml)

link_libraries(ggml)

add_executable(ggml-test main.cpp)
add_executable(simple-add simple_add.cpp)

