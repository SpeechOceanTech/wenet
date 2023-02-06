from ubuntu:20.04

WORKDIR /opt/wenet

CMD mkdir /data/models

ARG LIB_DIR=/lib/
COPY fc_base/libtorch-src/lib/libtorch_cpu.so $LIB_DIR
COPY fc_base/libtorch-src/lib/libtorch.so $LIB_DIR
COPY fc_base/libtorch-src/lib/libc10.so $LIB_DIR
COPY fc_base/libtorch-src/lib/libgomp-a34b3233.so.1 $LIB_DIR
COPY build/server/libwenet/libwenet.so $LIB_DIR

COPY server/model-server bin/

ENTRYPOINT ["bin/model-server"]