#!/bin/bash
rsync --exclude={'solutions/','results/','result_*'} -v -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    stdbuf -o 0 -e 0 ./generate_results.sh base 4 32768 32 32 | tee results/generate_results_out.txt;
    "
