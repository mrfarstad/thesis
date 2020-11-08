scp -r ./* yme:~/thesis_autotune
ssh -f yme "cd thesis_autotune; nohup ./autotune.sh yme laplace3d 2>&1 | tee results/out.txt &"
