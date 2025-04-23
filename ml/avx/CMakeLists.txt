cmake_minimum_required(VERSION 3.10)
project(avx512_array_add)

set(CMAKE_CXX_STANDARD 17)

# if (MSVC)
#     add_compile_options(/arch:AVX512)
# else()
#     add_compile_options(-mavx512f)
# endif()

add_executable(avx-test avx-test.cpp)