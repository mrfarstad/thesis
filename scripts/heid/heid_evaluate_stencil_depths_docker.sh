#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

working_folder='thesis'
sed -i -re 's/(YME_WORKING_FOLDER=).*/\1'$working_folder'/' $project_folder/constants.sh
source $project_folder/constants.sh

rsync --exclude={'solutions/','results/'} -v -r ./* heid:~/$YME_WORKING_FOLDER

container=stencils
image=martinrf/thesis
container_working_folder='..\/usr\/src\/thesis'

ssh heid -t "
    cd $YME_WORKING_FOLDER
    sed -i -re 's/(YME_WORKING_FOLDER=).*/\1$container_working_folder/' ./constants.sh
    source ./constants.sh
    bash ./scripts/evaluate_stencil_depths_docker.sh
    "
