#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
for d in "$@"
do
  :
  echo "Running CPU version (NX=NY=$d)"
  make -C $project_folder laplace2d_cpu DIM=$d
  $project_folder/bin/laplace2d_cpu
done
