scp -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    stdbuf -o 0 -e 0 ./run.sh base debug 32 32 yme | tee results/out.txt;
    "

 
