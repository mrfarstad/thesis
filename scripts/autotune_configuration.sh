#!/bin/bash

if [[ $# -lt 9 ]] ; then
    echo 'arg: VERSION NGPUS DIM DIMENSIONS STENCIL_DEPTH REPEAT SMEM_PAD UNROLL_X ITERATIONS'
    exit 0
fi

project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

config=$project_folder/configs/yme/general.conf
constants=$project_folder/constants.sh

bash $project_folder/scripts/set_run_configuration.sh $1 $2 $3 $4 $5 $7 $8 $9

source $constants

sed -i -re 's/(BLOCK_X = )[0-9|,| ]+/\11, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024/'     $config
sed -i -re 's/(BLOCK_Y = )[0-9|,| ]+/\11, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024/'     $config
if [ "$4" = "3" ] ; then
    sed -i -re 's/(BLOCK_Z = )[0-9|,| ]+/\11, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024/' $config
else
    sed -i -re 's/(BLOCK_Z = )[0-9|,| ]+/\11/'                                           $config
fi
sed -i -re 's/(repeat = )[0-9]*,/\1'$6',/'                                               $config
sed -i -re 's/(HEURISTIC = )[0-9]+/\10/'                                                 $config

# Extract all numerical results from run
# This output is used in in scripts/find_halo_depth.py

# Use when running on yme
#stdbuf -o 0 -e 0 python -u $project_folder/Autotuning/tuner/tune.py $config | tee /dev/tty | awk '/Version/{x=NR+1}(NR<=x){print}' | awk '$0==($0+0)'

# Use when running in Docker container on heid
stdbuf -o 0 -e 0 python2 -u $project_folder/Autotuning/tuner/tune.py $config | tee /dev/pts/0 | awk '/Minimal valuation/{x=NR+3}(NR<=x){print}'
