#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
if [[ $# -lt 2 ]] ; then
    echo 'arg: (prod/debug) HOST'
    exit 0
fi
[ ! -f $project_folder/solutions/solution\_$DIM\_$ITERATIONS ] && $project_folder/scripts/create_solutions.sh $DIM
#make -C $project_folder laplace2d_cpu DIM=$5
#$project_folder/bin/laplace2d_cpu
$project_folder/scripts/build.sh $@
$project_folder/bin/laplace2d_$1
#make -C $project_folder clean
