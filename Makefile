INC       := -I$(CUDA_HOME)/include -I.
LIB       := -L$(CUDA_HOME)/lib64  
LIBS      := -lcudart -lcudadevrt -Xcompiler -fopenmp
NVCCFLAGS := -lineinfo -rdc=true #--ptxas-options=-v #-Xptxas -dlcm=cg #--use_fast_math -lgomp #-arch=$(ARCH)
ifeq ($(DEBUG), true)
    #NVCC_DEBUG := -g -G # Causes troubles with floating point numbers if uncommented
    _DEBUG := -D DEBUG=true
endif
ifeq ($(HOST), yme)
    ARCH := sm_70
else ifeq ($(HOST), heid)
    ARCH := sm_70
else ifeq ($(HOST), idun)
    ARCH := sm_60
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
ifneq ($(STENCIL_DEPTH),)
    _STENCIL_DEPTH := -D STENCIL_DEPTH=$(STENCIL_DEPTH)
endif
ifneq ($(UNROLL_X),)
    _UNROLL_X := -D UNROLL_X=$(UNROLL_X)
endif
ifneq ($(DIMENSIONS),)
    _DIMENSIONS := -D DIMENSIONS=$(DIMENSIONS)
endif
ifneq ($(SMEM_PAD),)
    _SMEM_PAD := -D SMEM_PAD=$(SMEM_PAD)
endif
ifneq ($(PADDED),)
    _PADDED := -D PADDED=$(PADDED)
endif
ifneq ($(REGISTER),)
    _REGISTER := -D REGISTER=$(REGISTER)
endif
ifneq ($(HEURISTIC),)
    _HEURISTIC := -D HEURISTIC=$(HEURISTIC)
endif

all: 		stencil_$(ID)

stencil_$(ID): src/main.cu include/stencil_utils.h include/stencil_error_checker.h include/constants.h
	       nvcc src/main.cu -O3 -o bin/stencil_$(ID) -arch $(ARCH) \
		     $(NVCC_DEBUG) $(INC) $(LIB) $(NVCCFLAGS) $(LIBS)  \
		      			  -D BLOCK_X=$(BLOCK_X)        \
		      			  -D BLOCK_Y=$(BLOCK_Y)        \
		      			  -D BLOCK_Z=$(BLOCK_Z)        \
		      			  -D DIM=$(DIM)                \
	  $(_STENCIL_DEPTH) $(_SMEM) $(_COOP) $(_NGPUS) $(_ITERATIONS) \
	       $(_DIMENSIONS) $(_UNROLL_X) $(_UNROLL_DIM) $(_SMEM_PAD) \
		       $(_REGISTER) $(_PADDED) $(_HEURISTIC) $(_DEBUG)
					     

stencil_cpu:   include/stencil_initializer.h src/stencil_cpu.cu
		 gcc src/stencil_cpu.cpp -O3 -o bin/stencil_cpu -D DIM=$(DIM) $(_ITERATIONS)

profile:
	sudo ncu -f -o profile bin/stencil_$(ID)
		
clean:
		rm -f bin/stencil_*
		rm -f result solution

clean_results:
		rm -f results/*
