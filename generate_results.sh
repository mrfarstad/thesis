#!/bin/bash
#sizes=(256 512 1024 2048 4096, 8192, 16384, 32768)
sizes=(256 512) #1024 2048 4096, 8192, 16384, 32768)

for i in "${sizes[@]}"
do
  :
  echo "$1 (NX=NY=$i)"
  #perl -i -pe's/DIM=\d*/DIM=512/g' configs/hpclab13/$1.conf
  ./hpclab13_autotune.sh $1
  awk '/Minimal valuation/{x=NR+3}(NR<=x){print}' out.txt >> tst.txt
done


#stdbuf -o 0 -e 0 ./autotune.sh hpclab13 $1 laplace2d | tee results/out.txt
#awk '{if ($1=="rms" && $2=="error") print}' results/out.txt | tee results/errors.txt
