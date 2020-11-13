INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64
LIBS 	:= -lcudart -lcudadevrt
ifeq ($(BUILD), debug)
    NVCC_DEBUG := -g -G
    DEBUG := -D DEBUG=true
endif
ifeq ($(BUILD), test)
    TEST := -D TEST=true
endif
ifeq ($(HOST), yme)
    ARCH := sm_70
else
    ARCH := sm_75
endif

NVCCFLAGS	:= -lineinfo -rdc=true --ptxas-options=-v #--use_fast_math #-arch=$(ARCH) 

all: 		laplace2d_$(ID)

laplace2d_$(ID): solution laplace2d.cu laplace2d_kernel.cu laplace2d_utils.h laplace2d_error_checker.h Makefile
		 nvcc laplace2d.cu -o bin/laplace2d_$(ID) -arch $(ARCH)   \
		       $(NVCC_DEBUG) $(INC) $(LIB) $(NVCCFLAGS) $(LIBS)   \
						  -D BLOCK_X=$(BLOCK_X)   \
						  -D BLOCK_Y=$(BLOCK_Y)   \
						       $(DEBUG) $(TEST)  
							     

laplace2d_cpu:   laplace2d_initializer.h laplace2d_cpu_kernel.h
		 gcc laplace2d_cpu.cpp -o bin/laplace2d_cpu

profile:
	sudo ncu -f -o profile bin/laplace2d_$(ID)
		
clean:
		rm -f bin/laplace2d_*
		rm -f result solution

clean_results:
		rm -f results/*
