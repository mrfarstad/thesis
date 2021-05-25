#!/bin/bash
if [[ $# -lt 13 ]] ; then
    echo 'arg: VERSION NGPUS DIM DIMENSIONS HOST HEURISTIC BLOCK_X BLOCK_Y BLOCK_Z STENCIL_DEPTH SMEM_PAD UNROLL_X ITERATIONS'
    exit 0
fi

project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

constants=$project_folder/constants.sh

bash $project_folder/scripts/set_run_configuration.sh $1 $2 $3 $4 ${10} ${11} ${12} ${13}
sed -i -re 's/(HEURISTIC=)[0-1]+/\1'$6'/'   $constants
sed -i -re 's/(BLOCK_X=)[0-9|,| ]+/\1'$7'/' $constants
sed -i -re 's/(BLOCK_Y=)[0-9|,| ]+/\1'$8'/' $constants
sed -i -re 's/(BLOCK_Z=)[0-9|,| ]+/\1'$9'/' $constants
source $constants


rm -f profile.txt
bash $project_folder/scripts/build.sh profile $5 #> /dev/null
nvprof --metrics all --events all -f ./bin/stencil_profile 2>&1 | tee /dev/pts/0 | sed '1,/Kernel:/d; s/^ *//' | grep "^[0-9]" > profile.txt
