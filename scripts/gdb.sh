#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
$project_folder/scripts/build.sh debug yme
cuda-gdb $project_folder/bin/stencil_debug
