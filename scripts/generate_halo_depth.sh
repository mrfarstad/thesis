#!/bin/bash

if [[ $# -lt 6 ]] ; then
    echo 'arg: (base/smem/coop/coop_smem) NGPUS DIM BLOCK_X BLOCK_Y BLOCK_Z HALO_DEPTH'
    exit 0
fi

project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
config=$project_folder/configs/yme/deep_halo.conf
source $project_folder/constants.sh

# $1 is reserved for later implementations supporting (base/smem)
sed -i -re 's/(NGPUS = )[0-9]+/\1'$2'/' $config
sed -i -re 's/(DIM = )[0-9]+/\1'$3'/' $config
sed -i -re 's/(BLOCK_X = )[0-9|,| ]+/\1'$4'/' $config
sed -i -re 's/(BLOCK_Y = )[0-9|,| ]+/\1'$5'/' $config
sed -i -re 's/(BLOCK_Z = )[0-9|,| ]+/\1'$6'/' $config
sed -i -re 's/(HALO_DEPTH = )[0-9]+/\1'$7'/' $config
sed -i -re 's/(repeat = )[0-9]+/\1'$REPEAT'/' $config

# Extract all numerical results from run
# This output is used in in scripts/find_halo_depth.py
stdbuf -o 0 -e 0 python $project_folder/Autotuning/tuner/tune.py $config | tee /dev/tty | awk '/Version/{x=NR+1}(NR<=x){print}' | awk '$0==($0+0)'
