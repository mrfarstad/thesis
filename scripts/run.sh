#!/bin/bash
if [[ $# -lt 5 ]] ; then
    echo 'arg: (base/smem/coop/coop_smem) (prod/debug) BLOCK_X BLOCK_Y DIM HOST'
    exit 0
fi
iter=128
echo $5
[ ! -f solutions/solution\_$5\_$iter ] && $(dirname "$0")/create_solutions.sh $5
##make laplace2d_cpu DIM=$5
#./bin/laplace2d_cpu
$(dirname "$0")/build.sh $@
bin/laplace2d_$2
#make clean
