#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'arg: hpclab13/yme'
    exit 0
fi

iter=1024
#versions=(coop coop_smem)
#versions=(base smem)
versions=(base smem coop coop_smem)
sizes=(32 64 128 256 512 1024 2048 4096)
# 4096 8192)
#sizes=(32 64 128)
host=$1

#./create_solutions.sh ${sizes[@]}
for s in "${sizes[@]}"
do
  :
  [ ! -f solutions/solution\_$s\_$iter ] && ./create_solutions.sh $s
done

#rm -f tst.txt
for v in "${versions[@]}"
do
  :
    for s in "${sizes[@]}"
    do
      :
      echo "$v (NX=NY=$s 1 GPU)" >> results/1_gpu.txt
      sed -i -re 's/(DIM=)[0-9]+/\1'$s'/' configs/$host/$v.conf
      ###stdbuf -o 0 -e 0 ./autotune.sh $host $v laplace2d > results/out.txt
      wait
      stdbuf -o 0 -e 0 ./autotune.sh $host $v laplace2d | tee results/out_"$v"_"$s".txt
      wait
      # Only use this if DEBUG=true
      awk '{if ($1=="rms" && $2=="error") print}' results/out_"$v"_"$s".txt >> results/1_gpu_errors.txt
      awk '/Minimal valuation/{x=NR+3}(NR<=x){print}' results/out_"$v"_"$s".txt >> results/1_gpu.txt
    done
done
