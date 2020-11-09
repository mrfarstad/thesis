make ID=prod BLOCK_X="$1" BLOCK_Y="$2" HOST="$3" #BUILD=debug
./bin/laplace2d_prod
make clean
