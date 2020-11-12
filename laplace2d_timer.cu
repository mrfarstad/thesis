#include <stdio.h>

void stop_timer(cudaEvent_t *start, cudaEvent_t *stop, float *m, const char * format)
{
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);
    cudaEventElapsedTime(m, *start, *stop);
    printf(format, *m);
}
