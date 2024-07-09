
# 编译动态库libwenet.so
mkdir build
cmake -DONNX=ON -DTORCH=OFF -DWENET_LIB=ON -S . -B build
cmake --build build -j64

# 编译model-server
cd server;
go build .

# 构建Docker镜像
cd ..
TAG=$(date +%Y%m%d%H%M%S)
sudo docker build . -t wenet-runtime-onnx:$TAG -f onnx.dockerfile
sudo docker tag wenet-runtime-onnx:$TAG registry.cn-hangzhou.aliyuncs.com/speechocean/wenet-runtime:$TAG
