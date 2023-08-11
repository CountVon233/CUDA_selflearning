#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>

struct edge{
    int next;
    int adj;
};

__global__ void pull_iter(edge * A, double * result, double * next_result, int * degree, int num_vertice, double delta, int * e_begin) {
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    //int stride = (num_vertice + blockDim.x - 1) / blockDim.x;
    for(int j = e_begin[index]; j >= 0; j = A[j].next)
        next_result[index] += result[A[j].adj];
    next_result[index] = (delta * next_result[index] + (1.0 - delta) / (double) num_vertice) / (double) degree[index];
}

__global__ void swap_result(double * result, double * next_result, int num_vertice) {
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    //int stride = (num_vertice + blockDim.x - 1) / blockDim.x;

    //for (int i = 0; i < stride && index * stride + i < num_vertice ; ++i) {
        result[index] = next_result[index];
        next_result[index] = 0.0;
    //}
}

int cuda_main()
{
    std::fstream vertice_file;
    std::fstream edge_file;

    int num_vertice = 0;
    int num_edge = 0;
    int max_round = 1000;
    double delta = 0.85;
    
    vertice_file.open("../p2p-31.v", std::ios::in);
    if(vertice_file.is_open()){
        std::string s_v;
        while(std::getline(vertice_file, s_v)){
            int v = 0;
            int item = 1;
            for(int i = 0; i < s_v.size(); ++i){
                if(s_v[i] == ' '){
                    for(int j = i - 1; j >= 0; --j){
                        v += (s_v[j] - '0') * item;
                        item *= 10;
                    }
                    break;
                }  
            }
            num_vertice = (v > num_vertice)? v : num_vertice;
        }
    }
    vertice_file.close();

    edge_file.open("../p2p-31.e", std::ios::in);
    if(edge_file.is_open()){
        std::string s_e;
        while(std::getline(edge_file, s_e)){
            num_edge++;
        }
    }
    edge_file.close();
    
    //std::cout << num_vertice << std::endl;
    //std::cout << num_edge << std::endl;

    // 申请托管内存
    edge * A;
    int *degree, *e_begin;
    double *result, *next_result;
    cudaMallocManaged((void**)& A, 2 * num_edge * sizeof(edge));
    cudaMallocManaged((void**)& degree, num_vertice * sizeof(int));
    cudaMallocManaged((void**)& e_begin, num_vertice * sizeof(int));
    cudaMallocManaged((void**)& result, num_vertice * sizeof(double));
    cudaMallocManaged((void**)& next_result, num_vertice * sizeof(double));

    // 初始化数据
    for(int i = 0; i < num_vertice; ++i){
        result[i] = (1.0 - delta) / (double) num_vertice;
        next_result[i] = 0.0;
        degree[i] = 0;  
        e_begin[i] = -1;
    }
    for(int i = 0; i < num_edge; ++i){
        A[i].next = -1;
        A[i].adj = 0;
    }
        
    edge_file.open("../p2p-31.e", std::ios::in);
    int cnt = 0;
    if(edge_file.is_open()){
        std::string s_e;
        while(std::getline(edge_file, s_e)){
            int v = 0, u = 0;
            int item = 1;
            for(int i = 0; i < s_e.size(); ++i){
                if(s_e[i] == ' '){
                    for(int j = i + 1; j < s_e.size(); ++j){
                        if(s_e[j] == ' '){
                            for(int k = i - 1; k >= 0; --k){
                                v += (s_e[k] - '0') * item;
                                item *= 10;
                            }
                            item = 1;
                            for(int k = j - 1; k > i; --k){
                                u += (s_e[k] - '0') * item;
                                item *= 10;
                            }
                            break;
                        }
                    }
                    break;
                }
            }
            if(v < u){
                v--;
                u--;
                A[cnt].adj = u;
                A[cnt].next = e_begin[v];
                e_begin[v] = cnt;
                degree[v]++;
                cnt++;
                A[cnt].adj = v;
                A[cnt].next = e_begin[u];
                e_begin[u] = cnt;
                degree[u]++;
                cnt++;
            }
        }
    }
    edge_file.close();

    for(int i = 0; i < num_vertice; ++i){
        result[i] = result[i] / (double) degree[i];
    }
    
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((num_vertice + blockSize.x - 1) / blockSize.x);

    // 执行kernel
    while(max_round--){
        pull_iter <<< gridSize, blockSize >>>(A, result, next_result, degree, num_vertice, delta, e_begin);
        cudaDeviceSynchronize();
        swap_result <<< gridSize, blockSize >>>(result, next_result, num_vertice);
        cudaDeviceSynchronize();
    }
    
    // 检查执行结果
    for (int i = 0; i < num_vertice; i++)
        std::cout << i << ": " << result[i] * (double) degree[i] << std::endl;

    // 释放内存
    cudaFree(A);
    cudaFree(degree);
    cudaFree(result);
    cudaFree(next_result);

    return 0;
}