#!/bin/bash
if [[ $# -lt 11 ]] ; then
    echo 'arg: VERSION NGPUS DIM DIMENSIONS BLOCK_X BLOCK_Y BLOCK_Z STENCIL_DEPTH SMEM_PAD UNROLL_X ITERATIONS'
    exit 0
fi

project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

constants=$project_folder/constants.sh

bash $project_folder/scripts/set_run_configuration.sh $1 $2 $3 $4 $8 $9 ${10} ${11}
sed -i -re 's/(BLOCK_X=)[0-9|,| ]+/\1'$5'/' $constants
sed -i -re 's/(BLOCK_Y=)[0-9|,| ]+/\1'$6'/' $constants
sed -i -re 's/(BLOCK_Z=)[0-9|,| ]+/\1'$7'/' $constants
source $constants

gpu_index=0
rm -f profile.txt
bash $project_folder/scripts/set_cuda_visible_devices.sh $gpu_index
bash $project_folder/scripts/build.sh profile yme #> /dev/null
nvprof --metrics all --events all -f ./bin/stencil_profile 2>&1 | tee /dev/pts/0 | sed '1,/Kernel:/d; s/^ *//' | grep "^[0-9]" > profile.txt
