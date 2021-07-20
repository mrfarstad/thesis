#!/bin/bash
if [[ $# -lt 8 ]] ; then
    echo 'arg: (base/smem/smem_padded/coop) NGPUS DIM DIMENSIONS RADIUS SMEM_PAD COARSEN_X ITERATIONS'
    exit 0
fi
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
configuration_file=$project_folder/constants.sh
if [ "$1" = "base" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1false/'       $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'       $configuration_file
elif [[ $1 =~ "smem" ]]; then
  sed -i -re 's/(SMEM=)[a-z]+/\1true/'        $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'       $configuration_file
  sed -i -re 's/(PADDED=)[a-z]+/\1false/'     $configuration_file
  sed -i -re 's/(REGISTER=)[a-z]+/\1false/'   $configuration_file
  if [[ $1 =~ "smem_padded" ]]; then
    sed -i -re 's/(PADDED=)[a-z]+/\1true/'    $configuration_file
  elif [[ $1 =~ "smem_register" ]]; then
    sed -i -re 's/(REGISTER=)[a-z]+/\1true/'  $configuration_file
  fi 
elif [ "$1" = "coop" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1false/'       $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1true/'        $configuration_file
fi
sed -i -re 's/(NGPUS=)[0-9]+/\1'$2'/'         $configuration_file
sed -i -re 's/(DIM=)[0-9]+/\1'$3'/'           $configuration_file
sed -i -re 's/(DIMENSIONS=)[0-9]+/\1'$4'/'    $configuration_file
sed -i -re 's/(RADIUS=)[0-9]+/\1'$5'/' $configuration_file
sed -i -re 's/(SMEM_PAD=)[0-9]+/\1'$6'/'      $configuration_file
sed -i -re 's/(COARSEN_X=)[0-9]+/\1'$7'/'      $configuration_file
sed -i -re 's/(ITERATIONS=)[0-9]+/\1'$8'/'    $configuration_file
source $configuration_file
