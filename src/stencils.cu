#ifndef STENCILS_CU
#define STENCILS_CU

#include "../include/constants.h"

__host__ __device__ float stencil(float *d_u1, unsigned int idx) {
    unsigned int s, dim;
    int const offsets[] = {1, NX, NX*NY};
    //float const coef[] = {-1.0f, 1.0f/6.0f, 1.0f/6.0f};
    //float u = coef[0] * d_u1[idx];
    float c = (float) (6 * STENCIL_DEPTH);
    float u = 0.0f;
    for (dim=0; dim<3; dim++)
        for (s=1; s<=STENCIL_DEPTH; s++)
            u+=(d_u1[idx+s*offsets[dim]] + d_u1[idx-s*offsets[dim]]);
    return u / c - d_u1[idx];
}

#endif
