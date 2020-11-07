scp -r ./* yme:~/thesis_autotune
ssh yme -t "cd thesis_autotune; ./autotune.sh yme yme.conf"
