make laplace2d_cpu
./bin/laplace2d_cpu
./build.sh $1 debug 32 32 256 hpclab13
#make ID=debug BUILD=debug BLOCK_X=32 BLOCK_Y=32 HOST=hpclab13 && 
cuda-gdb ./bin/laplace2d_debug
