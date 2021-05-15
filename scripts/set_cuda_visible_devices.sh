#!/bin/bash
if [[ $# -lt 1 ]] ; then
    echo 'arg: CUDA_VISIBLE_DEVICES=[0-15]+[,0-15]*'
    exit 0
fi

project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

gpu_index=$1
# Use this if you require only a single GPU
set_single_gpu_visible() {
    sed -i -re "s/(export CUDA_VISIBLE_DEVICES=)[0-9,]+/\1$gpu_index/" $project_folder/constants.sh
}
# Use this if you require all GPUs
set_all_gpus_visible() {
    sed -i -re 's/(export CUDA_VISIBLE_DEVICES=)[0-9,]+//' $project_folder/constants.sh
}

if grep -q "CUDA_VISIBLE_DEVICES" $project_folder/constants.sh
then
    if [[ $gpu_index == all ]];then
        set_all_gpus_visible
    else
        set_single_gpu_visible
    fi
else
    if [[ ! $gpu_index == all ]];then
        echo "export CUDA_VISIBLE_DEVICES=$gpu_index" >> $project_folder/constants.sh
    fi
fi
