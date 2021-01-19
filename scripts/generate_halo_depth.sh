#!/bin/bash

if [[ $# -lt 5 ]] ; then
    echo 'arg: (base/smem/coop/coop_smem) NGPUS DIM BLOCK_X BLOCK_Y HALO_DEPTH'
    exit 0
fi

config=deep_halo.conf
repeat=20
iter=64
dim=32768

[ ! -f solutions/solution\_$dim\_$iter ] && ./create_solutions.sh $dim

sed -i -re 's/(NGPUS = )[0-9]+/\1'$2'/' $config
sed -i -re 's/(DIM = )[0-9]+/\1'$3'/' $config
sed -i -re 's/(BLOCK_X = )[0-9|,| ]+/\1'$4'/' $config
sed -i -re 's/(BLOCK_Y = )[0-9|,| ]+/\1'$5'/' $config
sed -i -re 's/(repeat = )[0-9]+/\1'$repeat'/' $config
sed -i -re 's/(HALO_DEPTH = )[0-9]+/\1'$6'/' $config

# Extract all numerical results from run
stdbuf -o 0 -e 0 python ${PWD}/Autotuning/tuner/tune.py $config | tee /dev/tty | awk '/Version/{x=NR+1}(NR<=x){print}' | awk '$0==($0+0)'
