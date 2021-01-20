#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'arg: base/smem/coop/coop_smem'
    exit 0
fi

#make laplace2d_cpu
#./bin/laplace2d_cpu
$(dirname "$0")/build.sh $1 debug 32 32 2048 hpclab13
#make ID=debug BUILD=debug BLOCK_X=32 BLOCK_Y=32 HOST=hpclab13 && 
cuda-gdb bin/laplace2d_debug
