#!/bin/bash
if [[ $# -lt 1 ]] ; then
    echo 'arg: (heuristic/autotune/profile)'
    exit 0
fi

project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')

# Remove nvprof command
sed -i -re '/RUN \/bin\/bash -c "source .\/constants.sh \&\& .\/scripts\/build.sh profile yme"/d' $project_folder/Dockerfile

if [[ $1 == heuristic ]];then
    sed -i -re 's/(ENTRYPOINT) \[.*\]/\1 \["python3", "-u", "\.\/scripts\/evaluate_stencil_depths\.py"\]/' $project_folder/Dockerfile
elif [[ $1 == autotune ]];then
    sed -i -re 's/(ENTRYPOINT) \[.*\]/\1 \["python3", "-u", "\.\/scripts\/evaluate_stencil_depths\.py", "True"\]/' $project_folder/Dockerfile
elif [[ $1 == profile ]];then
    #sed -i -re 's/(ENTRYPOINT) \[.*\]/RUN \/bin\/bash -c "source .\/constants.sh \&\& .\/scripts\/build.sh profile yme"\nENTRYPOINT ["nvprof", "--analysis-metrics", "-o", "bin\/profile.prof", "-f", ".\/bin\/stencil_profile"]/' $project_folder/Dockerfile
    sed -i -re 's/(ENTRYPOINT) \[.*\]/RUN \/bin\/bash -c "source .\/constants.sh \&\& .\/scripts\/build.sh profile yme"\nENTRYPOINT ["nvprof", "--metrics", "dram_read_throughput,dram_utilization,gld_transactions,gst_transactions,gld_efficiency,gst_efficiency", "-f", ".\/bin\/stencil_profile"]/' $project_folder/Dockerfile
fi
