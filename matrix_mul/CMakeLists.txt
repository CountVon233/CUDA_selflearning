cmake_minimum_required(VERSION 3.22)
project(matrix_mul LANGUAGES CXX CUDA)

add_executable(matrix_mul
    main.cpp
    matrix_mul.cu
)