#!/bin/bash
if [[ $# -lt 1 ]] ; then
    echo 'arg: CUDA_VISIBLE_DEVICES=[0-15]+[,0-15]*'
    exit 0
fi

project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

gpu_index=$1
if grep -q "CUDA_VISIBLE_DEVICES" $project_folder/constants.sh
then
    sed -i -re "s/(export CUDA_VISIBLE_DEVICES=)[0-9,]+/\1$gpu_index/" $project_folder/constants.sh
else
    if [[ ! $gpu_index == all ]];then
        echo "export CUDA_VISIBLE_DEVICES=$gpu_index" >> $project_folder/constants.sh
    fi
fi
