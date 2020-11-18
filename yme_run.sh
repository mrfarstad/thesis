rsync -v -r ./* yme:~/thesis
ssh yme -t 'cd thesis; ./run.sh smem debug 32 32 2048 yme'


