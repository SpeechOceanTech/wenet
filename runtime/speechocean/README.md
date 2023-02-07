# Wenet Runtime实战

## Torch版本

构建

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

## ONNX版本

构建

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

导出ONNX模型

```bash
# ONNX_MODEL_DIR为ONNX模型目录，MODEL_DIR是模型目录
PYTHONPATH=. python -m wenet.bin.export_onnx_cpu \
  --config $MODEL_DIR/train.yaml \
  --checkpoint $MODEL_DIR/final.pt \
  --chunk_size -1 \
  --output_dir $ONNX_MODEL_DIR \
  --num_decoding_left_chunks -1
```

解码

```bash
# wenet/runtime/speechocean/build 目录下运行
# ONNX_MODEL_DIR为导出ONNX模型目录
bin/decoder_main --chunk_size -1 --wav_path ~/Downloads/zh-cn-demo.wav --onnx_dir $ONNX_MODEL_DIR --unit_path $ONNX_MODEL_DIR/words.txt
```

运行容器

```bash
docker run --rm -p8080:8080 -v $ONNX_MODEL_DIR:/data/models wenet-runtime-onnx -model_path /data/models/ -model_name wenet -model_version 1.0
```

