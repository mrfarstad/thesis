#!/bin/bash
build=prod
rsync --exclude={'solutions/','results/','result_*'} -v -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    ./build.sh $1 $build 32 32 32768 yme;
    sudo $(which nvprof) ./bin/laplace2d_$build
    "
rsync -v -r yme:~/thesis_autotune/profile .

#    sudo $(which nvprof) -o profile -f ./bin/laplace2d_debug
