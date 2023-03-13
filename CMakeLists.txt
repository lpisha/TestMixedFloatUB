cmake_minimum_required(VERSION 3.9)
cmake_policy(SET CMP0104 OLD)
project(TestMixedFloatUB LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(TestMixedFloatUB test.cu)
target_include_directories(TestMixedFloatUB PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
