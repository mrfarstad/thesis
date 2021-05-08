#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

gpu_index=12
bash $project_folder/scripts/set_cuda_visible_devices.sh $gpu_index
bash $project_folder/scripts/configure_dockerfile.sh profile
bash $project_folder/scripts/heid/heid_docker.sh profile
