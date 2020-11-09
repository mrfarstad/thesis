make ID=prod BLOCK_X="$1" BLOCK_Y="$2" #BUILD=debug
./bin/laplace3d_prod
make clean
