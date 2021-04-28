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
#stdbuf -e 0 -o 0
#nvprof --log-file profile.txt --metrics dram_read_throughput,dram_utilization,dram_write_throughput,dram_write_transactions,gld_transactions,gst_transactions,gld_efficiency,gst_efficiency,gld_throughput,gst_throughput -f ./bin/stencil_profile
#nvprof --metrics dram_read_throughput,dram_utilization,dram_write_throughput,dram_write_transactions,gld_transactions,gst_transactions,gld_efficiency,gst_efficiency,gld_throughput,gst_throughput -f ./bin/stencil_profile 2>&1 | tee /dev/pts/0 > profile.txt
nvprof --metrics all -f ./bin/stencil_profile 2>&1 | tee /dev/pts/0 > profile.txt
sed -i -e '1,/Kernel:/d' profile.txt
awk '1;/thesis_profile/{exit}' profile.txt
#head -n-1 profile.txt > profile.txt
#cat profile.txt | sed -e '1,/Kernel:/d' | awk '1;/thesis_profile/{exit}' | head -n-1 > profile.txt
