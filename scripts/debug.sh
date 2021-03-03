#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
make -C $project_folder stencil_cpu
$project_folder/bin/stencil_cpu
$project_folder/scripts/build.sh debug hpclab13
$project_folder/bin/stencil_debug
make -C $project_folder clean
