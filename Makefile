FLAGS := -arch=sm_75 -rdc=true --ptxas-options=-v 
LINKER := -lcudadevrt
DEPS := common.h utils.h
CU_APPS := multi-gpu simple2DFD double

.PHONY: run clean

all: ${CU_APPS}

%: %.cu
    ifeq ($(BUILD), debug)
	nvcc -g -G ${FLAGS} -o $@ $< ${LINKER}
    else
	nvcc ${FLAGS} -o $@ $< ${LINKER}
    endif

clean:
	rm -Rf *.o
	rm -Rf ${CU_APPS}
