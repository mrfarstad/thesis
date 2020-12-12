#!/bin/bash
rsync --exclude={'solutions/','results/','result_*'} -v -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    python3 find_halo_depth.py
    "
