#!/bin/bash
#sizes=(256 512 1024 2048 4096, 8192, 16384, 32768)
#versions=(base smem coop coop_smem)
sizes=(256 512)
versions=(base smem)

for i in "${versions[@]}"
do
  :
    for s in "${sizes[@]}"
    do
      :
      echo "$i (NX=NY=$s)" >> tst.txt
      perl -i -pe"s/DIM=\d*/DIM=$s/g" configs/hpclab13/$i.conf
      ./hpclab13_autotune.sh $i
      awk '/Minimal valuation/{x=NR+3}(NR<=x){print}' results/out.txt >> tst.txt
    done
done


#stdbuf -o 0 -e 0 ./autotune.sh hpclab13 $1 laplace2d | tee results/out.txt
#awk '{if ($1=="rms" && $2=="error") print}' results/out.txt | tee results/errors.txt
