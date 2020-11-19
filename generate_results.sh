#!/bin/bash

######## THOUGHTS ##########
#This script should take the domain dim and the block dims as parameters.
#This is because it is the python script who has generated the optimal parameters, and therefore the script will provide them.

if [[ $# -lt 3 ]] ; then
    echo 'arg: DIM BLOCK_X BLOCK_Y'
    exit 0
fi

config=smem.conf
iter=1024
[ ! -f solutions/solution\_$1\_$iter ] && ./create_solutions.sh $1
wait
sed -i -re 's/(DIM = )[0-9]+/\1'$1'/' $config
sed -i -re 's/(BLOCK_X = )[0-9]+/\1'$2'/' $config
sed -i -re 's/(BLOCK_Y = )[0-9]+/\1'$3'/' $config
python ${PWD}/Autotuning/tuner/tune.py $config
