#ifndef STENCILS_CU
#define STENCILS_CU

#include "../include/constants.h"

__host__ __device__ float stencil(float *d_u1, unsigned int idx) {
    float u = 0.0f, c = (float) (2*DIMENSIONS*STENCIL_DEPTH);
    unsigned int dim, s;
    unsigned int offsets[DIMENSIONS] = {1};
#if DIMENSIONS>1
    offsets[1] = NX;
#endif
#if DIMENSIONS>2
    offsets[2] = NX*NY;
#endif
    for (dim=0; dim<DIMENSIONS; dim++)
        for (s=1; s<=STENCIL_DEPTH; s++)
            u+=(d_u1[idx-s*offsets[dim]] + d_u1[idx+s*offsets[dim]]);
    return u / c - d_u1[idx];
}

#endif
