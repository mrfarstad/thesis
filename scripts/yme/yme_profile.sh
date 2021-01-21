#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
build=prod
rsync --exclude={'solutions/','results/'} -v -r ./* yme:~/$YME_WORKING_FOLDER
ssh yme -t "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh
    ./build.sh $build yme;
    sudo $(which nvprof) ./bin/laplace2d_$build
    "
rsync -v -r yme:~/$YME_WORKING_FOLDER/profile .
#    sudo $(which nvprof) -o profile -f ./bin/laplace2d_debug
#
