FLAGS := -arch=sm_75
DEPS := common.h utils.h

.PHONY: multi-gpu run clean

multi-gpu: $(DEPS) multi-gpu.cu
	nvcc $(FLAGS) multi-gpu.cu -o multi-gpu

run:
	$(MAKE) --no-print-directory
	./multi-gpu
	$(MAKE) clean --no-print-directory

clean:
	rm -Rf *.o
	rm -Rf multi-gpu 

# end
