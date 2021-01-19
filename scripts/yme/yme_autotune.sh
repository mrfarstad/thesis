rsync -r ./* yme:~/thesis_autotune

ssh yme -t "
    cd thesis_autotune;
    stdbuf -o 0 -e 0 ./autotune.sh yme $1 laplace2d | tee results/out.txt;
    awk '{if (\$1==\"rms\" && \$2==\"error\") print}' results/out.txt | tee results/errors.txt
    "
    #./autotune.sh yme $1 laplace2d > results/out.txt;
    #./find_optimal_block_size.sh yme
