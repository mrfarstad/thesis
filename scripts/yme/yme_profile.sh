#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
build=profile
output=bin/profile.prof
rsync --exclude={'solutions/','results/'} -v -r ./* yme:~/$YME_WORKING_FOLDER
ssh yme -t "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh
    ./scripts/build.sh $build yme;
    sudo $(which nvprof) -o $output -f ./bin/laplace3d_$build
    "
rsync -v -r yme:~/$YME_WORKING_FOLDER/$output bin/
#    sudo $(which nvprof) -o profile -f ./bin/laplace2d_debug
#
