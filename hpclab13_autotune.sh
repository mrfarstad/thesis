stdbuf -o 0 -e 0 ./autotune.sh hpclab13 laplace2d | tee results/out.txt
awk '{if ($2=="error") print}' results/out.txt | tee results/errors.txt
