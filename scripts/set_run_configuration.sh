#!/bin/bash
if [[ $# -lt 3 ]] ; then
    echo 'arg: (base/smem/coop/coop_smem) NGPUS HALO_DEPTH STENCIL_DEPTH'
    exit 0
fi
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
configuration_file=$project_folder/constants.sh
if [ "$1" = "base" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1false/'    $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'    $configuration_file
elif [ "$1" = "smem" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1true/'     $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'    $configuration_file
elif [ "$1" = "coop" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1false/'    $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1true/'     $configuration_file
elif [ "$1" = "coop_smem" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1true/'     $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1true/'     $configuration_file
fi
sed -i -re 's/(NGPUS=)[0-9]+/\1'$2'/'      $configuration_file
sed -i -re 's/(STENCIL_DEPTH=)[0-9]+/\1'$3'/' $configuration_file
#sed -i -re 's/(HALO_DEPTH=)[0-9]+/\1'$3'/' $configuration_file
source $configuration_file
