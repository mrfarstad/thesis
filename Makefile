FLAGS := -arch=sm_75
DEPS := common.h utils.h
CU_APPS := multi-gpu simple2DFD

.PHONY: run clean

all: ${CU_APPS}

%: %.cu
	nvcc ${FLAGS} -o $@ $<

clean:
	rm -Rf *.o
	rm -Rf ${CU_APPS}
