1、远程登录主机
在BASIC-5G WIFI下
```bash
$ ssh countvon@192.168.0.160 -p 12345
```
在SJTU WIFI下
```bash
$ ssh countvon@202.120.38.217 -p 12345
```
在校外，先登录SJTUvpn，然后按SJTU WIFI下方法登录

vscode远程登录：先在.ssh目录下的config文件中配置新的Host
```bash
Host [hostname]
    HostName 202.120.38.217
    User countvon
    Port 12345
```
然后在vscode远程登录中选择"Connect to Host..."并通过配置好的Host访问主机

2、编译、运行CUDA文件
```bash
$ nvcc [filename].cu -o [filename]
$ [filename]
```

3、学习资源

快速入门 

https://zhuanlan.zhihu.com/p/34587739

https://zhuanlan.zhihu.com/p/97044592

nvidia官方CUDA C编程指导手册中文版（主要学习使用资料） 

https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese/tree/main

潭升CUDA博客（答疑参考资料） 

https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89

nvidia官方CUDA C++编程指导手册（可结合中文学习资料使用）

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

nvidia官方CUDA C++最佳实践指导（进阶内容，主要用于提升性能，后期有帮助）

https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

英文参考书（大概率用不到）

https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf

4、Application List

vector_add : 向量加法

matrix_mul : 矩阵乘法

matrix_vector_mul : 矩阵向量乘

5、通过CMake编译、运行Application（以vector_add为例）：
```bash
$ cd vector_add/build
$ cmake .. 
$ make
$ ./vector_add
```