FLAGS := -arch=sm_75

.PHONY: multi-gpu clean

multi-gpu: common/common.h multi-gpu.cu
	nvcc $(FLAGS) multi-gpu.cu -o multi-gpu

run:
	$(MAKE) --no-print-directory
	./multi-gpu
	$(MAKE) clean --no-print-directory

clean:
	rm -Rf *.o
	rm -Rf multi-gpu 

# end
