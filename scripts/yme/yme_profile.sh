#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
build=profile
output=bin/profile.prof
rsync --exclude={'solutions/','results/'} -v -r ./* yme:~/$YME_WORKING_FOLDER
ssh yme -t "
    cd $YME_WORKING_FOLDER;
    source ./constants.sh
    ./scripts/build.sh $build yme;
    sudo \$(which nvprof) --print-gpu-trace -f ./bin/stencil_$build;
    "

    #sudo \$(which nvprof) --analysis-metrics -o $output -f ./bin/stencil_$build;
    #sudo \$(which nvprof) --metrics dram_read_throughput,dram_utilization,gld_transactions,gst_transactions,gld_efficiency,gst_efficiency -f ./bin/stencil_$build;

rsync -v -r yme:~/$YME_WORKING_FOLDER/$output bin/
