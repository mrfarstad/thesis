#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
rsync --exclude={'solutions/','results/'} -r ./* yme:~/$YME_WORKING_FOLDER
ssh yme -t "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh;
    stdbuf -o 0 -e 0 ./scripts/generate_results.sh base $NGPUS $DIM $BLOCK_X $BLOCK_Y $BLOCK_Z;
    "
# | tee results/generate_results_out.txt;
