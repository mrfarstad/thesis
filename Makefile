INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64
LIBS 	:= -lcudart -lcudadevrt
ARCH    := sm_75
ifeq ($(BUILD), debug)
    DEBUG := -g -G
endif

NVCCFLAGS	:= -lineinfo -rdc=true --ptxas-options=-v --use_fast_math #-arch=$(ARCH) 

all: 		laplace2d_$(ID)

laplace2d_$(ID): laplace2d.cu laplace2d_gold.cpp laplace2d_kernel.cu Makefile
		 nvcc laplace2d.cu laplace2d_gold.cpp -o bin/laplace2d_$(ID) \
		      $(DEBUG) $(INC) $(LIB) $(NVCCFLAGS) $(LIBS)            \
		 			    -D BLOCK_X=$(BLOCK_X)            \
		 			    -D BLOCK_Y=$(BLOCK_Y)            
profile:
	sudo ncu -f -o profile bin/laplace2d_$(ID)
		
clean:
		rm -f bin/laplace2d_*

clean_results:
		rm -f results/*
