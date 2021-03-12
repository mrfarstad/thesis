#ifndef STENCILS_CU
#define STENCILS_CU

#include "../include/constants.h"

__host__ __device__ float stencil(float *d_u1, unsigned int idx) {
    unsigned int const offsets[] = {1, NX, NX*NY};
    float u = 0.0f;
#if SQUARE_STENCIL
    float u0;
    unsigned int i, j, k, sidx;
    for (k=(-STENCIL_DEPTH); k=<STENCIL_DEPTH; k++) {
        for (j=(-STENCIL_DEPTH); j=<STENCIL_DEPTH; j++) {
            for (i=(-STENCIL_DEPTH); i=<STENCIL_DEPTH; i++) {
                if (k==0 && j==0 && i==0) { u0 = d_u1[idx]; continue; }
                u+=d_u1[idx + i + j*NX + k*NX*NY];
            }
        }
    }
    return u / (float) (3*(2*STENCIL_DEPTH+1)-1) - u0;
#else
    unsigned char dim, s;
    for (dim=0; dim<3; dim++)
        for (s=1; s<=STENCIL_DEPTH; s++)
            u+=(d_u1[idx-s*offsets[dim]] + d_u1[idx+s*offsets[dim]]);
    return u / (float) (6 * STENCIL_DEPTH) - d_u1[idx];
#endif
}

#endif
