INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_75 --ptxas-options=-v --use_fast_math

all: 		laplace3d 

laplace3d:	laplace3d.cu laplace3d_gold.cpp laplace3d_kernel.h Makefile
		nvcc laplace3d.cu laplace3d_gold.cpp -o laplace3d \
				       $(INC) $(NVCCFLAGS) $(LIB) \
					    -D BLOCK_X=$(BLOCK_X) \
					    -D BLOCK_Y=$(BLOCK_Y) \
					    -D BLOCK_Z=$(BLOCK_Z)
clean:
		rm -f laplace3d
