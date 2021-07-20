#!/bin/bash

if [[ $# -lt 14 ]] ; then
    echo 'arg: VERSION NGPUS DIM DIMENSIONS HOST HEURISTIC BLOCK_X BLOCK_Y BLOCK_Z RADIUS REPEAT SMEM_PAD COARSEN_X ITERATIONS'
    exit 0
fi

project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

config=$project_folder/configs/yme/general.conf
constants=$project_folder/constants.sh

bash $project_folder/scripts/set_run_configuration.sh $1 $2 $3 $4 ${10} ${12} ${13} ${14}
source $constants

sed -i -re 's/(HOST=)[a-z]+/\1'$5'/'          $config
sed -i -re 's/(HEURISTIC = )[0-1]+/\1'$6'/'   $config
sed -i -re 's/(BLOCK_X = )[0-9|,| ]+/\1'$7'/' $config
sed -i -re 's/(BLOCK_Y = )[0-9|,| ]+/\1'$8'/' $config
sed -i -re 's/(BLOCK_Z = )[0-9|,| ]+/\1'$9'/' $config
sed -i -re 's/(repeat = )[0-9]+,/\1'${11}',/' $config

# Extract all numerical results from run
# This output is used in in scripts/find_halo_depth.py

# Use when running on yme
#stdbuf -o 0 -e 0 python -u $project_folder/Autotuning/tuner/tune.py $config | tee /dev/tty | awk '/Version/{x=NR+1}(NR<=x){print}' | awk '$0==($0+0)'

# Use when running on idun or heid
if [[ $5 == idun ]]; then
    #pts=$(ls -l /dev/pts/ | grep martinrf | awk -F" " '{print $NF}')
    out=thesis_output.out
else
    out=/dev/pts/0
fi
stdbuf -o 0 -e 0 python2 -u $project_folder/Autotuning/tuner/tune.py $config | tee -a $out | awk '/Version/{x=NR+1}(NR<=x){print}' | awk '$0==($0+0)'
