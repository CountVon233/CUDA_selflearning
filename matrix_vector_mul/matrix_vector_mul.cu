#include <iostream>

__global__ void MatVecMul(float *x, float *y, float *z, int n) {
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = (n + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < stride && index * stride + i < n ; ++i) {
       for(int j = 0; j < n; ++j)
            z[index * stride + i] += x[(index * stride + i) * n + j] * y[j];
    }
}

int cuda_main()
{
    int N = 1 << 10;

    // 申请托管内存
    float *x, *y, *z;
    cudaMallocManaged((void**)&x, N * N * sizeof(float));
    cudaMallocManaged((void**)&y, N * sizeof(float));
    cudaMallocManaged((void**)&z, N * sizeof(float));

    // 初始化数据
    for (int i = 0; i < N * N; ++i){
        x[i] = 1.0;
    }
    for (int i = 0; i < N; ++i){
        y[i] = 1.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // 执行kernel
    MatVecMul <<< gridSize, blockSize >>>(x, y, z, N);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    
    // 检查执行结果
    for (int i = 0; i < N; i++)
        std::cout << i << ": " << z[i] << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}