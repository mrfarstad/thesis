#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../include/laplace2d_utils.h"
#include "../include/laplace2d_initializer.h"
#include "../include/laplace2d_cpu_kernel.h"

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, const char **argv){
    unsigned long    i,
           ibyte = NX*NY * sizeof(float);
    float  *h_u1, *h_u3, *h_swap;
    double start, elapsed;

    h_u1 = (float *)malloc(ibyte);
    h_u3 = (float *)malloc(ibyte);

    initialize_host_region(h_u1);

    start = seconds();
    for (i = 1; i <= ITERATIONS; ++i) {
        cpu_laplace2d(h_u1, h_u3);
        h_swap = h_u1; h_u1 = h_u3; h_u3 = h_swap;   // swap h_u1 and h_u3
    }
    elapsed = seconds() - start;
    printf("\ncpu_laplace2d: %.3f (ms) \n", elapsed);

    saveSolution(h_u1);

    free(h_u1);
    free(h_u3);
}
