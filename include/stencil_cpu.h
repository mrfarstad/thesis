#ifndef STENCIL_CPU
#define STENCIL_CPU

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "constants.h"
#include "stencil_initializer.h"
#include "stencil_cpu_kernel.h"
#include "stencil_utils.h"

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void stencil_cpu()
{
    unsigned int    i;
    float  *h_u1, *h_u3, *h_swap;
    double start, elapsed;

    h_u1 = (float *)malloc(BYTES);
    h_u3 = (float *)malloc(BYTES);

    initialize_host_region(h_u1);

    start = seconds();
    for (i = 1; i <= ITERATIONS; ++i) {
        cpu_stencil(h_u1, h_u3);
        h_swap = h_u1; h_u1 = h_u3; h_u3 = h_swap;   // swap h_u1 and h_u3
    }
    elapsed = seconds() - start;
    printf("\ncpu_stencil: %.3f (ms) \n", elapsed);

    saveSolution(h_u1);

    free(h_u1);
    free(h_u3);
}

#endif
