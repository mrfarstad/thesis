FROM nvidia/cuda:11.1-devel

RUN apt-get update -y
RUN apt-get install build-essential -y --no-install-recommends

WORKDIR /usr/src/laplace3d

COPY *.sh *.cu *.cpp *.h Makefile ./

CMD ./run.sh 32 16 2
