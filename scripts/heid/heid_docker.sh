#!/bin/bash
if [[ $# -lt 1 ]] ; then
    echo 'arg: (stencil_depths/unroll)'
    exit 0
fi
rsync --exclude={'solutions/'} -v -r ./* heid:~/
ssh minip -t "
    srun -N1 -n1 -c16 --gres=gpu:16 --partition=HEID -w heid --time=1:00:00 --pty /bin/bash ./scripts/docker.sh
"
rsync -v heid:~/results.json results/results_$1.json

