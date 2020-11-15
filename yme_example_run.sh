scp -r ./* yme:~/thesis
ssh yme -t "
    cd thesis;
    stdbuf -o 0 -e 0 ./run.sh coop debug 32 32 yme | tee results/out.txt;
    "

 
