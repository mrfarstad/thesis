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

__device__ float stencil(float smem[BLOCK_Z+2*STENCIL_DEPTH][BLOCK_Y+2*STENCIL_DEPTH][BLOCK_X+2*STENCIL_DEPTH],
                         int sz, int sy, int sx) {
    unsigned int s;
    float u = 0.0f, c = (float) (6 * STENCIL_DEPTH);
    for (s=1; s<=STENCIL_DEPTH; s++)
        //u+=(smem[sz][sy][sx-s] + smem[sz][sy][sx+s]);
        u+=(smem[sz][sy][sx+s] + smem[sz][sy][sx-s]);
    for (s=1; s<=STENCIL_DEPTH; s++)
        //u+=(smem[sz][sy-s][sx] + smem[sz][sy+s][sx]);
        u+=(smem[sz][sy+s][sx] + smem[sz][sy-s][sx]);
    for (s=1; s<=STENCIL_DEPTH; s++)
        u+=(smem[sz+s][sy][sx] + smem[sz-s][sy][sx]);
    return u / c - smem[sz][sy][sx];
}

#endif
