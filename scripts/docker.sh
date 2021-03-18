#!/bin/bash
sed -i -re 's/(YME_WORKING_FOLDER=).*/\1..\/usr\/src\/thesis/' ./constants.sh
source ./constants.sh
container=thesis
image=martinrf/thesis
if [[ "$(docker images -q $image)" == "" ]]; then
    nvidia-docker rmi $image -f
fi
nvidia-docker build . -t $image
nvidia-docker run -it --name $container $image
nvidia-docker cp $container:/usr/src/thesis/results.json .
nvidia-docker rm $container -f
