FROM nvidia/cuda:11.1-devel

RUN apt-get update && apt-get install build-essential -y --no-install-recommends

WORKDIR /usr/src/laplace3d

COPY simple2DFD.sh simple2DFD.cu Makefile common.h utils.h ./

CMD ./simple2DFD.sh
