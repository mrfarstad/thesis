#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'arg: base/smem/coop/coop_smem'
    exit 0
fi

make laplace2d_cpu
./bin/laplace2d_cpu
#make BLOCK_X=32 BLOCK_Y=2 BUILD=debug ID=debug 
./build.sh $1 debug 32 32 2048 hpclab13
./bin/laplace2d_debug
make clean
