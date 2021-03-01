#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
host=yme
out_path=results/out.txt
gpus=(1 2 4)

for g in "${gpus[@]}"
do
  :
    if [[ $g -eq 1 ]] ; then
        versions=(base smem coop coop_smem)
        halo_depths=(1)
    else
        versions=(base smem)
        halo_depths=(1 2 4)
    fi
    for v in "${versions[@]}"
    do
      :
        for h in "${halo_depths[@]}"
        do
          :
          $project_folder/scripts/set_run_configuration.sh $v $g $h
          $project_folder/scripts/run.sh prod yme | tee ${out_path}
          error=$(awk '/reading solution/{getline;print;}' ${out_path})
          if [[ ! -z $(echo "$error" | awk '!/rms error = 0.000000/') ]] ; then
              echo "#############################"
              echo "ERROR"
              echo "$g GPU[s] $v HALO_DEPTH=$h"
              echo "$error"
              echo "#############################"
              exit
          fi
          rm ${out_path}
        done
    done
done

echo "#############################"
echo "CONGRATULATIONS!"
echo "NO ERRORS IN TESTED CONFIGURATIONS"
echo "#############################"
exit 0
