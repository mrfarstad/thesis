#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
make -C $project_folder ID="$1" BUILD="$1" HOST="$2"
