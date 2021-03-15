#!/bin/bash
container=thesis
image=martinrf/thesis
if [[ "$(docker images -q $image)" == "" ]]; then
    nvidia-docker rmi $image -f
fi
nvidia-docker build . -t $image
nvidia-docker run -it --name $container $image
nvidia-docker cp $container:/usr/src/thesis/results_stencil_depths.json .
nvidia-docker rm $container
