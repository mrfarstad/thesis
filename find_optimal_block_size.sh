#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'arg: hpclab13/yme'
    exit 0
fi

iter=1024
#versions=(base smem coop coop_smem)
versions=(coop coop_smem)
#sizes=(32 64 128 256 512 1024 2048 4096 8192)
sizes=(32 64 128)
host=$1

#./create_solutions.sh ${sizes[@]}
for s in "${sizes[@]}"
do
  :
  [ ! -f solutions/solution\_$s\_$iter ] && ./create_solutions.sh $s
done

rm -f tst.txt
for v in "${versions[@]}"
do
  :
    for s in "${sizes[@]}"
    do
      :
      echo "$v (NX=NY=$s)" >> tst.txt
      sed -i -re 's/(DIM=)[0-9]+/\1'$s'/' configs/$1/$v.conf
      ###stdbuf -o 0 -e 0 ./autotune.sh hpclab13 $v laplace2d > results/out.txt
      wait
      stdbuf -o 0 -e 0 ./autotune.sh hpclab13 $v laplace2d | tee results/out_"$v"_"$s".txt
      wait
      awk '/Minimal valuation/{x=NR+3}(NR<=x){print}' results/out_"$v"_"$s".txt >> tst.txt
    done
done
