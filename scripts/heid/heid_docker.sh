#!/bin/bash
if [[ $# -lt 1 ]] ; then
    echo 'arg: (stencil_depths/unroll)'
    exit 0
fi
rsync --exclude={'solutions/'} -v -r --delete . heid:~/thesis
ssh minip -t "
    srun -N1 -n1 -c1 --gres=gpu:1 --partition=HEID -w heid --time=0 --pty /bin/bash ./thesis/scripts/docker.sh
"
rsync -v heid:~/thesis/results.json results/results_$1.json

