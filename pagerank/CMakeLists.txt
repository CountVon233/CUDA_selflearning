cmake_minimum_required(VERSION 3.22)
project(pagerank LANGUAGES CXX CUDA)

add_executable(pagerank
    main.cpp
    pagerank.cu
)