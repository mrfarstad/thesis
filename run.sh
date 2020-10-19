#! /bin/sh
# rebuild prog if necessary
make simple2DFD
# run prog with some arguments
./simple2DFD "$@"
