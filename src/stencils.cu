#ifndef STENCILS_CU
#define STENCILS_CU

#include "../include/constants.h"

__host__ __device__ float stencil(float *d_u1, unsigned int idx) {
#if DIMENSIONS==3
    unsigned int const offsets[] = {1, NX, NX*NY};
#else
    unsigned int const offsets[] = {1, NX};
#endif
    float u = 0.0f, c = (float) (2*DIMENSIONS*STENCIL_DEPTH);
    unsigned int dim, s;
    for (dim=0; dim<DIMENSIONS; dim++)
        for (s=1; s<=STENCIL_DEPTH; s++)
            u+=(d_u1[idx-s*offsets[dim]] + d_u1[idx+s*offsets[dim]]);
    return u / c - d_u1[idx];
}

#endif
