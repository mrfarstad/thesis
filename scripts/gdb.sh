#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
#make -C $project_folder laplace3d_cpu
#$project_folder/bin/laplace3d_cpu
$project_folder/scripts/build.sh debug yme
#make -C $project_folder ID=debug BUILD=debug BLOCK_X=32 BLOCK_Y=32 HOST=hpclab13 && 
cuda-gdb $project_folder/bin/laplace3d_debug
