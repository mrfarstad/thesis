#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
rsync --exclude={'solutions/','results/','venv'} -r ./* yme:~/$YME_WORKING_FOLDER
ssh yme -t "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh
    stdbuf -o 0 -e 0 ./scripts/run.sh prod yme rm | tee results/out.txt;
    "
