#! /bin/sh
# rebuild prog if necessary
make simpleMultiGPU
# run prog with some arguments
./simpleMultiGPU "$@"
