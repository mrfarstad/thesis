#!/bin/bash
#if [[ ! -z $(awk '/rms error/{x=NR+1}(NR<=x){print $4}' ${path}_errors.txt | awk '!/0.000000/') ]] ; then
out=$(awk '/reading solution/{getline;print;}' out.txt)
echo $out
#if [[ $out == 'rms error = 0.000392' ]] ; then
#    echo "it works"
#fi
