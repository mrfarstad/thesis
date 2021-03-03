#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
rsync --exclude={'solutions/','results/'} -v -r ./* yme:~/$YME_WORKING_FOLDER
ssh yme -t "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh
    docker build . -t nvidia-test;
    "
    #docker create -it --gpus all --name stencil nvidia-test bash;
    #docker cp stencil:/usr/src/laplace2d/results/laplace2d.png .
