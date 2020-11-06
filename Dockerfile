FROM nvidia/cuda:11.1-devel

RUN apt-get update -y && DEBIAN_FRONTEND="noninteractive" apt-get install python lsof gnuplot build-essential -y --no-install-recommends

WORKDIR /usr/src/laplace3d

COPY Autotuning ./Autotuning
COPY *.sh *.cu *.conf Makefile ./

CMD ./autotune.sh laplace3d
