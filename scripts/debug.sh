#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
make -C $project_folder laplace3d_cpu
$project_folder/bin/laplace3d_cpu
$project_folder/scripts/build.sh debug hpclab13
$project_folder/bin/laplace3d_debug
make -C $project_folder clean
