#!/bin/bash
for d in "$@"
do
  :
  echo "Running CPU version (NX=NY=$d)"
  make laplace2d_cpu DIM=$d
  bin/laplace2d_cpu
done
