INC       := -I$(CUDA_HOME)/include -I.
LIB       := -L$(CUDA_HOME)/lib64  
LIBS      := -lcudart -lcudadevrt -Xcompiler -fopenmp
NVCCFLAGS := -lineinfo -rdc=true --use_fast_math #-lgomp  #--ptxas-options=-v #-arch=$(ARCH)
ifeq ($(DEBUG), true)
    NVCC_DEBUG := -g -G
    _DEBUG := -D DEBUG=true
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
ifneq ($(HALO_DEPTH),)
    _HALO_DEPTH := -D HALO_DEPTH=$(HALO_DEPTH)
endif

all: 		laplace3d_$(ID)

laplace3d_$(ID): src/laplace3d.cu src/laplace3d_kernel.cu include/laplace3d_utils.h include/laplace3d_error_checker.h include/constants.h
		 nvcc src/laplace3d.cu -O3 -o bin/laplace3d_$(ID) -arch $(ARCH)   \
		       $(NVCC_DEBUG) $(INC) $(LIB) $(NVCCFLAGS) $(LIBS)   \
						  -D BLOCK_X=$(BLOCK_X)   \
						  -D BLOCK_Y=$(BLOCK_Y)   \
						  -D BLOCK_Z=$(BLOCK_Z)   \
						  -D DIM=$(DIM)           \
			     $(_SMEM) $(_COOP) $(_NGPUS) $(_HALO_DEPTH)   \
							      $(_DEBUG)  
							     

laplace3d_cpu:   include/laplace3d_initializer.h include/laplace3d_cpu_kernel.h
		 gcc src/laplace3d_cpu.cpp -O3 -o bin/laplace3d_cpu -D DIM=$(DIM) $(_ITERATIONS)

profile:
	sudo ncu -f -o profile bin/laplace3d_$(ID)
		
clean:
		rm -f bin/laplace3d_*
		rm -f result solution

clean_results:
		rm -f results/*
