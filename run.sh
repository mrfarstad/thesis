make ID=prod BLOCK_X="$1" BLOCK_Y="$2" BLOCK_Z="$3"
./bin/laplace3d_prod
make clean
