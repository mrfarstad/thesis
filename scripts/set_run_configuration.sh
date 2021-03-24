#!/bin/bash
if [[ $# -lt 5 ]] ; then
    echo 'arg: (base/smem/coop/coop_smem) NGPUS DIM DIMENSIONS STENCIL_DEPTH SMEM_PAD UNROLL_X'
    exit 0
fi
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
configuration_file=$project_folder/constants.sh
if [ "$1" = "base" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(UNROLL_X=)[0-9]+/\1'1'/'                 $configuration_file
elif [ "$1" = "base_unroll_2" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(UNROLL_X=)[0-9]+/\1'2'/'                 $configuration_file
elif [ "$1" = "smem" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1true/'                    $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(PREFETCH=)[a-z]+/\1false/'               $configuration_file
  sed -i -re 's/(UNROLL_X=)[0-9]+/\1'1'/'                 $configuration_file
elif [ "$1" = "smem_prefetch" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1true/'                    $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(PREFETCH=)[a-z]+/\1true/'               $configuration_file
  sed -i -re 's/(UNROLL_X=)[0-9]+/\1'1'/'                 $configuration_file
elif [ "$1" = "smem_unroll_4" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1true/'                    $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(PREFETCH=)[a-z]+/\1false/'               $configuration_file
  sed -i -re 's/(UNROLL_X=)[0-9]+/\1'4'/'                 $configuration_file
elif [ "$1" = "smem_prefetch_unroll_4" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1true/'                    $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(UNROLL_X=)[0-9]+/\1'4'/'                 $configuration_file
  sed -i -re 's/(PREFETCH=)[a-z]+/\1true/'                $configuration_file
elif [ "$1" = "coop" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1false/'                   $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1true/'                    $configuration_file
elif [ "$1" = "coop_smem" ] ; then
  sed -i -re 's/(SMEM=)[a-z]+/\1true/'                    $configuration_file
  sed -i -re 's/(COOP=)[a-z]+/\1true/'                    $configuration_file
fi
sed -i -re 's/(NGPUS=)[0-9]+/\1'$2'/'                     $configuration_file
sed -i -re 's/(DIM=)[0-9]+/\1'$3'/'                       $configuration_file
sed -i -re 's/(DIMENSIONS=)[0-9]+/\1'$4'/'                $configuration_file
sed -i -re 's/(STENCIL_DEPTH=)[0-9]+/\1'$5'/'             $configuration_file
[ ! -z "$6" ] && sed -i -re 's/(SMEM_PAD=)[0-9]+/\1'$6'/' $configuration_file
[ ! -z "$7" ] && sed -i -re 's/(UNROLL_X=)[0-9]+/\1'$7'/' $configuration_file
source $configuration_file
