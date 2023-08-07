cmake_minimum_required(VERSION 3.22)
project(matrix_vector_mul LANGUAGES CXX CUDA)

add_executable(matrix_vector_mul
    main.cpp
    matrix_vector_mul.cu
)