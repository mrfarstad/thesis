#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
sed -i -re 's/(scripts\/).+(\.py)/\1evaluate_unroll\2/' $project_folder/Dockerfile
bash $project_folder/scripts/heid/heid_docker.sh unroll
