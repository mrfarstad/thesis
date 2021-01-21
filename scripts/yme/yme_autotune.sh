#!/bin/bash
if [[ $# -lt 1 ]] ; then
    echo 'arg: (base/smem/coop/coop_smem)'
    exit 0
fi
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
rsync --exclude={'solutions/','results/'} -v -r ./* yme:~/$YME_WORKING_FOLDER
ssh yme -t "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh
    stdbuf -o 0 -e 0 ./autotune.sh yme $1 | tee results/out.txt;
    awk '{if (\$1==\"rms\" && \$2==\"error\") print}' results/out.txt | tee results/errors.txt
    "
    #./autotune.sh yme $1 laplace2d > results/out.txt;
    #./find_optimal_block_size.sh yme
