make laplace2d_cpu
./bin/laplace2d_cpu
#make BLOCK_X=32 BLOCK_Y=2 BUILD=debug ID=debug 
./build.sh $1 debug 32 32 hpclab13
./bin/laplace2d_debug
make clean
