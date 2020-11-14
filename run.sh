#!/bin/bash
make laplace2d_cpu
./bin/laplace2d_cpu
./build.sh $@
./bin/laplace2d_$2
#make clean
