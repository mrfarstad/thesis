FROM nvidia/cuda:11.1-devel

RUN apt-get update
RUN apt-get -y install python2
RUN apt-get -y install python3

WORKDIR /usr/src/thesis

COPY . ./

RUN /bin/bash -c "source ./constants.sh && ./scripts/build.sh profile yme"
ENTRYPOINT ["nvprof", "-o", "bin/profile.prof", "-f", "./bin/stencil_profile"]
