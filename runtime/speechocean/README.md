# Wenet Runtime实战

## Torch版本

### 构建

1. 编译动态库`libwenet.so`

```bash
# wenet/runtime/speechocean目录下运行
mkdir build
cmake -DTORCH=ON -DWENET_LIB=ON -S . -B build
cmake --build build -j64
```

> -DWENET_BIN构建二进制文件`wenet-runtime`，-DWENET_LIB=ON构建动态库`libwenet.so`被集成。

2. 编译`model-server`

```bash
# wenet/runtime/speechocean目录下运行
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

> 镜像大小633MB

## ONNX版本

### 构建

1. 编译动态库`libwenet.so`

```bash
# wenet/runtime/speechocean目录下运行
mkdir build
cmake -DONNX=ON -DTORCH=OFF -DWENET_LIB=ON -S . -B build
cmake --build build -j64
```

> -DWENET_BIN构建二进制文件`wenet-runtime`，-DWENET_LIB=ON构建动态库`libwenet.so`被集成。

2. 编译`model-server`

```bash
# wenet/runtime/speechocean目录下运行
cd server;
go build .
```

3. 构建镜像

```bash
# wenet/runtime/speechocean目录下运行
TAG=$(date +%Y%m%d%H%M%S)
docker build . -t wenet-runtime-onnx:$TAG -f onnx.dockerfile
docker tag wenet-runtime-onnx:$TAG registry.cn-hangzhou.aliyuncs.com/speechocean/wenet-runtime:$TAG
```

> onnx去除了libtorch_cpu.so的依赖，镜像大小为123MB



### 导出ONNX模型

```bash
# ONNX_MODEL_DIR为ONNX模型目录，MODEL_DIR是模型目录
PYTHONPATH=. python -m wenet.bin.export_onnx_cpu \
  --config $MODEL_DIR/train.yaml \
  --checkpoint $MODEL_DIR/final.pt \
  --chunk_size -1 \
  --output_dir $ONNX_MODEL_DIR \
  --num_decoding_left_chunks -1
```

### 解码

```abash
# wenet/runtime/speechocean/build 目录下运行
# ONNX_MODEL_DIR为导出ONNX模型目录
bin/decoder_main --chunk_size -1 --wav_path ~/Downloads/zh-cn-demo.wav --onnx_dir $ONNX_MODEL_DIR --unit_path $ONNX_MODEL_DIR/words.txt
```

### 运行容器服务

```bash
docker run --rm -p8080:8080 -v $ONNX_MODEL_DIR:/data/models wenet-runtime-onnx -model_path /data/models/ -model_name wenet -model_version 1.0
```

测试

```bash
curl localhost:8080/infer -d '{"inputs":[{"wav_path": "http://so-algorithm-test.oss-cn-beijing.aliyuncs.com/samples/asr/zh-cn-02.wav"}]}'
```

输出

```json
{
  "model_name": "asr",
  "model_version": "1.0",
  "text": "甚至出现交易几乎停滞的情况"
}
```



## Runtime和Python版本对比

### 镜像大小对比

| Python版本 | Torch版本 | ONNX版本 |
| ---------- | --------- | -------- |
| 10.1GB     | 634MB     | 123MB    |

ONNX版本镜像主要包括：

- 基础镜像，Ubuntu20.04，大小为72.8MB
- 主要的动态库，libwenet.so 大小为23M，libonnxruntime.so 大小为15M
- 二进制文件，model-server 大小为11M

### 推理速度对比

TODO