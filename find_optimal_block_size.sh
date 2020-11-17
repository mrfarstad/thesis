#!/bin/bash

#sizes=(256 512 1024 2048 4096, 8192, 16384, 32768)
#sizes=(256 512)
#versions=(base smem)
#sizes=(256 512 1024 2048 4096 8192)
versions=(base smem coop coop_smem)
sizes=(32 64)
#versions=(base smem coop coop_smem)
rm -f tst.txt
for v in "${versions[@]}"
do
  :
    for s in "${sizes[@]}"
    do
      :
      echo "$v (NX=NY=$s)" >> tst.txt
      #perl -i -pe"s/DIM=\d*/DIM=$s/g" configs/hpclab13/$i.conf
      #sed -i -re 's/(DIM=)[0-9]+/\1'$s'/' configs/yme/$v.conf
      sed -i -re 's/(DIM=)[0-9]+/\1'$s'/' configs/hpclab13/$v.conf
      ./hpclab13_autotune.sh $v
      #./yme_autotune.sh $v
      awk '/Minimal valuation/{x=NR+3}(NR<=x){print}' results/out.txt >> tst.txt
    done
done


#stdbuf -o 0 -e 0 ./autotune.sh hpclab13 $1 laplace2d | tee results/out.txt
#awk '{if ($1=="rms" && $2=="error") print}' results/out.txt | tee results/errors.txt
