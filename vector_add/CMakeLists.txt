cmake_minimum_required(VERSION 3.22)
project(vector_add LANGUAGES CXX CUDA)

add_executable(vector_add
    main.cpp
    vector_add.cu
)