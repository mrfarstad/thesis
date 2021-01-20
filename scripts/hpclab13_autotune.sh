./autotune.sh hpclab13 $1 laplace2d > results/out.txt
awk '{if ($1=="rms" && $2=="error") print}' results/out.txt > reusults/errors.txt
#stdbuf -o 0 -e 0 ./autotune.sh hpclab13 $1 laplace2d | tee results/out.txt
