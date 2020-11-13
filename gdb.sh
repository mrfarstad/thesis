make laplace2d_cpu
./bin/laplace2d_cpu
make BLOCK_X=32 BLOCK_Y=2 BUILD=debug ID=debug 
cuda-gdb ./bin/laplace2d_debug
make clean
