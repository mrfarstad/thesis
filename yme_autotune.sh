scp -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    stdbuf -o 0 -e 0 ./autotune.sh yme laplace2d | tee results/out.txt
    awk '{if ($2=="error") print}' results/out.txt | tee results/errors.txt
    "
