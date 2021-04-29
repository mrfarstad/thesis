FROM nvidia/cuda:11.1-devel

RUN apt-get update
RUN apt-get -y install python2
RUN apt-get -y install python3

WORKDIR /usr/src/thesis

COPY . ./

ENTRYPOINT ["python3", "-u", "./scripts/evaluate_stencil_depths.py", "True", "True"]
