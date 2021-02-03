#!/bin/bash

#sizes=(256) # DEBUG
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
sizes=(256 512 1024 2048 4096 8192 16384 32768) # OLD
#sizes=(2048 4096 8192 16384 32768 65536)
gpus=(1 2 4)
host=yme

for g in "${gpus[@]}"
do
  :
    if [[ $g -eq 1 ]] ; then
        versions=(base smem coop coop_smem)
        #versions=(base) # DEBUG
        path=$project_folder/results/1_gpu
    else
        versions=(base smem)
        #versions=(base) # DEBUG
        path=$project_folder/results/${g}_gpus
    fi
    [ -f ${path}.txt ] && mv ${path}.txt ${path}_backup.txt
    for v in "${versions[@]}"
    do
      :
        for s in "${sizes[@]}"
        do
          :
          out_path=$project_folder/results/out_"$g"_"$v"_"$s".txt
          conf_path=$project_folder/configs/yme/$v.conf
          echo "conf_path: $conf_path"
          echo "$v" >> $path.txt
          sed -i -re 's/(NGPUS = )[0-9]+/\1'$g'/' $conf_path
          sed -i -re 's/(DIM = )[0-9]+/\1'$s'/' $conf_path
          #sed -i -re 's/(repeat = )[0-9]+/\1'1'/' $conf_path # DEBUG
          sed -i -re 's/(repeat = )[0-9]+/\1'$REPEAT'/' $conf_path
          #sed -i -re 's/(BLOCK_X =) .+/\1 32/' $conf_path # DEBUG
          #sed -i -re 's/(BLOCK_Y =) .+/\1 32/' $conf_path # DEBUG
          sed -i -re 's/(BLOCK_X =) .+/\1 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024/' $conf_path
          sed -i -re 's/(BLOCK_Y =) .+/\1 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024/' $conf_path
          stdbuf -o 0 -e 0 $project_folder/scripts/autotune.sh $host $v laplace3d | tee $out_path
          awk '{if ($1=="rms" && $2=="error") print}' $out_path > ${path}_errors.txt
          #awk '/rms error/{x=NR+1}(NR<=x){print $4}' 1_gpu_errors.txt | awk '!/0.000000/' # OLD
          #error=$(awk '/reading solution/{getline;print;}' ${out_path})
          #if [[ ! -z $(echo "$error" | awk '!/rms error = 0.000000/') ]] ; then
          #    echo "#############################"
          #    echo "ERROR"
          #    echo "$g GPU[s] $v DIM=$s ITERATIONS=$iter"
          #    echo "$error"
          #    echo "#############################"
          #    exit 0
          #fi
          #exit 0 # DEBUG
          awk '/Minimal valuation/{x=NR+3}(NR<=x){print}' $out_path >> $path.txt
        done
    done
done
