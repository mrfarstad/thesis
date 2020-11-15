scp -r ./* yme:~/thesis
ssh yme -t "
    cd thesis;
    stdbuf -o 0 -e 0 ./run.sh $1 debug 32 32 yme | tee results/out.txt;
    "

 
