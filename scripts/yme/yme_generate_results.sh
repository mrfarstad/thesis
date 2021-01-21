#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
rsync --exclude={'solutions/','results/'} -v -r ./* yme:~/$YME_WORKING_FOLDER
ssh yme -t "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh
    stdbuf -o 0 -e 0 ./generate_results.sh smem 1 8192 32 32;
    "
# | tee results/generate_results_out.txt;
