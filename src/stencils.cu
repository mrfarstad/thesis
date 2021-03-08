#ifndef STENCILS_CU
#define STENCILS_CU

#include "../include/constants.h"

__host__ __device__ float stencil(float *d_u1, unsigned int idx, unsigned int const offsets[3]) {
    float u = 0.0f, c = (float) (6 * STENCIL_DEPTH);
    for (unsigned int dim=0; dim<3; dim++)
        for (unsigned int s=1; s<=STENCIL_DEPTH; s++)
            u+=(d_u1[idx-s*offsets[dim]] + d_u1[idx+s*offsets[dim]]);
    return u / c - d_u1[idx];
}

__host__ __device__ float stencil(float *d_u1, unsigned int idx) {
    unsigned int const offsets[] = {1, NX, NX*NY};
    return stencil(d_u1, idx, offsets);
}

__host__ __device__ float smem_stencil(float *d_u1, int sz, int sy, int sx) {
    unsigned int const offsets[] = {1, BLOCK_X+2*STENCIL_DEPTH, (BLOCK_X+STENCIL_DEPTH*2)*(BLOCK_Y+STENCIL_DEPTH*2)};
    unsigned int idx = sx + (sy*offsets[1]) + (sz*offsets[2]);
    return stencil(d_u1, idx, offsets);
}

#endif
