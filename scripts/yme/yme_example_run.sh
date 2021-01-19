#!/bin/bash
rsync --exclude={'solutions/','results/','result_*'} -v -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    stdbuf -o 0 -e 0 ./run.sh base prod 32 32 32768 yme | tee results/out.txt;
    "
