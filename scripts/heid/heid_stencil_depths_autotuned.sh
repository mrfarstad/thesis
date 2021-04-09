#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
sed -i -re 's/(CUDA_VISIBLE_DEVICES=)[0-9]+/\11/' $project_folder/constants.sh
sed -i -re 's/(scripts\/).+(\.py).*/\1evaluate_stencil_depths\2", "True"]/' $project_folder/Dockerfile
bash $project_folder/scripts/heid/heid_docker.sh stencil_depths_autotuned
