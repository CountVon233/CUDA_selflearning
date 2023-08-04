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
Host [any hostname on yourself]
    HostName 202.120.38.217
    User countvon
    Port 12345
```
然后再vscode远程登录中选择"Connect to Host..."并通过配置好的Host访问主机
2、编译、运行CUDA文件
```bash
$ nvcc [filename].cu -o [filename]
$ [filename]
```
