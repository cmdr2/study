cmake_minimum_required(VERSION 3.12)
project(ggml-test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set(GGML_CUDA ON)

add_subdirectory(ggml)

link_libraries(ggml)

add_executable(simple-add simple_addition.cpp)
add_executable(simple-add-with-static-weights simple_addition_with_static_weights.cpp)
add_executable(logic-gate logic_gate.cpp)
add_executable(logic-gate-refactored logic_gate_refactored.cpp)
add_executable(logic-gate-with-weights-file logic_gate_with_weights_file.cpp safetensors.hpp json.hpp)

add_executable(safetensor-viewer safetensor_viewer.cpp safetensors.hpp json.hpp)
add_executable(large-add large_add.cpp)
