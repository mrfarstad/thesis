#!/bin/bash
project_folder=$(echo $(dirname "$0") | sed 's/thesis.*/thesis/')
config=$project_folder/deep_halo.conf
source $project_folder/constants.sh

sed -i -re 's/(NGPUS = )[0-9]+/\1'$2'/' $config
sed -i -re 's/(DIM = )[0-9]+/\1'$3'/' $config
sed -i -re 's/(BLOCK_X = )[0-9|,| ]+/\1'$4'/' $config
sed -i -re 's/(BLOCK_Y = )[0-9|,| ]+/\1'$5'/' $config
sed -i -re 's/(repeat = )[0-9]+/\1'$REPEAT'/' $config
sed -i -re 's/(HALO_DEPTH = )[0-9]+/\1'$6'/' $config

# Extract all numerical results from run
stdbuf -o 0 -e 0 python $project_folder/Autotuning/tuner/tune.py $config | tee /dev/tty | awk '/Version/{x=NR+1}(NR<=x){print}' | awk '$0==($0+0)'
