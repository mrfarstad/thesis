#!/bin/bash
rsync --exclude={'solutions/','results/','result_*'} -v -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    stdbuf -o 0 -e 0 python3 -u handle_results.py | tee results/generate_results_out.txt;
    "
