scp -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    ./autotune.sh yme $1 laplace2d > results/out.txt;
    "
    #awk '{if (\$1==\"rms\" && \$2==\"error\") print}' results/out.txt | tee results/errors.txt
    #stdbuf -o 0 -e 0 ./autotune.sh yme $1 laplace2d | tee results/out.txt;
