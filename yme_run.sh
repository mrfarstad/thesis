scp -r ./* yme:~/thesis
ssh yme -t 'cd thesis; ./run.sh base prod 16 8 yme'
