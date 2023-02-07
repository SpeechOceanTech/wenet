from ubuntu:20.04

WORKDIR /opt/wenet

CMD mkdir /data/models

ARG LIB_DIR=/lib/
COPY fc_base/onnxruntime-src/lib/libonnxruntime.so.1.12.0 $LIB_DIR
COPY build/server/libwenet/libwenet.so $LIB_DIR

COPY server/model-server bin/

ENV GLOG_logtostderr=1
ENV GLOG_v=2

ENTRYPOINT ["bin/model-server"]