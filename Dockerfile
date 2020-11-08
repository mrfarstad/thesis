FROM nvidia/cuda:11.1-devel

RUN apt-get update -y && DEBIAN_FRONTEND="noninteractive" apt-get install python lsof gnuplot build-essential -y --no-install-recommends

WORKDIR /usr/src/laplace3d

COPY Autotuning ./Autotuning
COPY bin/ ./bin
COPY results/ ./results
COPY *.sh *.cu *.conf *.h *.cpp Makefile ./

CMD ./autotune.sh yme laplace3d
