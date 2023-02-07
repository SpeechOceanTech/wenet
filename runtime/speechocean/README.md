# Wenet Runtime部署海天平台


## 构建

1. 编译动态库`libwenet.so`

```bash
mkdir build; 
cmake -DONNX=ON -DTORCH=ON -DWENET_BIN=ON -S . -B build/; 
cmake --build build -j64;
```

> -DWENET_BIN构建二进制文件`wenet-runtime`，-DWENET_LIB=ON构建动态库`libwenet.so`被集成。

2. 编译`model-server`

```
cd server;
go build .
```

3. 构建镜像

```bash
# wenet/runtime/speechocean目录下运行
TAG=$(date +%Y%m%d%H%M%S)
docker build . -t wenet-runtime:$TAG
docker tag wenet-runtime:$TAG registry.cn-hangzhou.aliyuncs.com/speechocean/wenet-runtime:$TAG
```

