#INC	:= -I$(CUDA_HOME)/include -I.
#LIB	:= -L$(CUDA_HOME)/lib64
#LIBS 	:= -lcudart -lcudadevrt
ARCH    := sm_75
ifeq ($(BUILD), debug)
    DEBUG := -g -G
endif

#NVCCFLAGS	:= -lineinfo -rdc=true --ptxas-options=-v --use_fast_math #-arch=$(ARCH) 

all: 		laplace3d_$(ID)

laplace3d_$(ID): laplace3d.cu laplace3d_gold.cpp laplace3d_kernel.cu Makefile
		 nvcc laplace3d.cu laplace3d_gold.cpp -o bin/laplace3d_$(ID) \
		      $(DEBUG) $(INC) $(LIB) $(NVCCFLAGS) $(LIBS)            \
		 			    -D BLOCK_X=$(BLOCK_X)            \
		 			    -D BLOCK_Y=$(BLOCK_Y)            \
		 			    -D BLOCK_Z=$(BLOCK_Z)
profile:
	sudo ncu -f -o profile bin/laplace3d_$(ID)
		
clean:
		rm -f bin/laplace3d_*

results_clean:
		rm -f results/*
