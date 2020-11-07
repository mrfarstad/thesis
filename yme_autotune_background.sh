scp -r ./* yme:~/thesis_autotune
ssh -f yme "cd thesis_autotune; nohup ./autotune.sh yme yme.conf > /dev/null 2>&1 &"
