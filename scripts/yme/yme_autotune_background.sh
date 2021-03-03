#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
rsync --exclude={'solutions/','results/'} -v -r ./* yme:~/$YME_WORKING_FOLDER
ssh -f yme "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh
    nohup ./autotune.sh yme stencil 2>&1 | tee results/out.txt &
    "
