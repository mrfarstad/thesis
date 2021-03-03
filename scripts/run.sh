#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
if [[ $# -lt 2 ]] ; then
    echo 'arg: (prod/debug) HOST'
    exit 0
fi
#make -C $project_folder stencil_cpu DIM=$5
#$project_folder/bin/stencil_cpu
if [ ! -z "$3" ]
  then
    rm -rf $project_folder/solutions/*
fi
$project_folder/scripts/build.sh $@
$project_folder/bin/stencil_$1
#make -C $project_folder clean
