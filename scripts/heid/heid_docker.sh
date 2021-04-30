#!/bin/bash
if [[ $# -lt 1 ]] ; then
    echo 'arg: (stencil_depths_heuristic/unroll/autotune/stencil_depths_autotuned/profile/batch_profile)'
    exit 0
fi
rsync --exclude={'solutions/','venv/'} -r --delete . heid:~/thesis_$1
# Slurm
#ssh minip -t "
#    srun -N1 -n1 -c1 --gres=gpu:1 --partition=HEID -w heid --time=0 --pty /bin/bash ./thesis_$1/scripts/docker.sh $1
#"

run_docker() {
    ssh heid -t "./thesis_$1/scripts/docker.sh $1"
}

run_docker $1
if [[ $1 == profile ]];then
    rsync -v heid:~/thesis_$1/bin/profile.prof bin/profile.prof
else
    rsync -v heid:~/thesis_$1/results.json results/results_$1.json
fi


