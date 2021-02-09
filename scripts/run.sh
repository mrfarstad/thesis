#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
if [[ $# -lt 2 ]] ; then
    echo 'arg: (prod/debug) HOST'
    exit 0
fi
#make -C $project_folder laplace3d_cpu DIM=$5
#$project_folder/bin/laplace3d_cpu
$project_folder/scripts/build.sh $@
$project_folder/bin/laplace3d_$1
#make -C $project_folder clean