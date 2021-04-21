#!/bin/bash
rsync --exclude={'solutions/'} -v -r --delete . idun:~/thesis
#ssh idun -t "cd thesis; sbatch job.slurm"
