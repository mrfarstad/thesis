INC       := -I$(CUDA_HOME)/include -I.
LIB       := -L$(CUDA_HOME)/lib64  
LIBS      := -lcudart -lcudadevrt -Xcompiler -fopenmp
NVCCFLAGS := -lineinfo -rdc=true --use_fast_math #-lgomp  #--ptxas-options=-v #-arch=$(ARCH)
ifeq ($(BUILD), debug)
    NVCC_DEBUG := -g -G
    DEBUG := -D DEBUG=true
endif
ifeq ($(BUILD), test)
    DEBUG := -D DEBUG=true
endif
ifeq ($(HOST), yme)
    ARCH := sm_70
else
    ARCH := sm_75
endif
ifeq ($(SMEM), true)
    _SMEM := -D SMEM=true
endif
ifeq ($(COOP), true)
    _COOP := -D COOP=true
endif
ifneq ($(DIM),)
    _DIM := -D DIM=$(DIM)
endif
ifneq ($(ITERATIONS),)
    _ITERATIONS := -D ITERATIONS=$(ITERATIONS)
endif
ifneq ($(NGPUS),)
    _NGPUS := -D NGPUS=$(NGPUS)
endif

all: 		laplace2d_$(ID)

laplace2d_$(ID): laplace2d.cu laplace2d_kernel.cu laplace2d_utils.h laplace2d_error_checker.h Makefile #solution
		 nvcc laplace2d.cu -o bin/laplace2d_$(ID) -arch $(ARCH)   \
		       $(NVCC_DEBUG) $(INC) $(LIB) $(NVCCFLAGS) $(LIBS)   \
						  -D BLOCK_X=$(BLOCK_X)   \
						  -D BLOCK_Y=$(BLOCK_Y)   \
						  -D DIM=$(DIM)           \
					    $(_SMEM) $(_COOP) $(_NGPUS)   \
							       $(DEBUG)  
							     

laplace2d_cpu:   laplace2d_initializer.h laplace2d_cpu_kernel.h
		 gcc laplace2d_cpu.cpp -o bin/laplace2d_cpu $(_DIM) $(_ITERATIONS)

profile:
	sudo ncu -f -o profile bin/laplace2d_$(ID)
		
clean:
		rm -f bin/laplace2d_*
		rm -f result solution

clean_results:
		rm -f results/*
