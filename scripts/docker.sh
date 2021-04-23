#!/bin/bash
if [[ $# -lt 1 ]] ; then
    echo 'arg: (stencil_depths_heuristic/unroll/autotune/stencil_depths_autotuned/profile)'
    exit 0
fi

run_container () {
  nvidia-docker run --privileged -it --name $container $image
}

delete_container () {
  nvidia-docker rm $container -f
}

cd thesis_$1
sed -i -re 's/(YME_WORKING_FOLDER=).*/\1..\/usr\/src\/thesis/' ./constants.sh
source ./constants.sh
container=thesis_$1
image=martinrf/thesis/$1
if [[ "$(docker images -q $image)" == "" ]]; then
    nvidia-docker rmi $image -f
fi
nvidia-docker build . -t $image
if ! run_container; then
    delete_container
    run_container
fi
if [[ $1 == profile ]];then
    nvidia-docker cp $container:/usr/src/thesis/bin/profile.prof bin/profile.prof
else
    nvidia-docker cp $container:/usr/src/thesis/results.json .
fi
delete_container
