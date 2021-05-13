#ifndef STENCILS_CU
#define STENCILS_CU

#include "../include/constants.h"
#include "stencils_border_check.cu"

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


__device__ __inline__ void accumulate_l(float *u, float *smem, unsigned int sidx, int soffset)
{
#pragma unroll
    for (unsigned int d=1; d<=STENCIL_DEPTH; d++) *u += smem[sidx-d*soffset];
}

__device__ __inline__ void accumulate_l(float *u, float *smem, float *d_u1, unsigned int sidx,
                                        unsigned int idx, unsigned int t,
                                        int soffset, int offset)
{
#pragma unroll
    for (unsigned int d=STENCIL_DEPTH; d>=1; d--) *u += (t >= d) ? smem[sidx-d*soffset] : d_u1[idx-d*offset];
}

__device__ __inline__ void accumulate_reg_l(float *u, float *yval)
{
#pragma unroll
    for (unsigned int d = 0; d < STENCIL_DEPTH; d++) *u += yval[d];
}

__device__ __inline__ void accumulate_reg_r(float *u, float *yval)
{
#pragma unroll
    for (unsigned int d = STENCIL_DEPTH+1; d < 2*STENCIL_DEPTH+1; d++) *u += yval[d];
}

__device__ __inline__ void accumulate_r(float *u, float *smem, unsigned int sidx, int soffset)
{
#pragma unroll
    for (unsigned int d=1; d<=STENCIL_DEPTH; d++) *u += smem[sidx+d*soffset];
}

__device__ __inline__ void accumulate_r(float *u, float *smem, float *d_u1, unsigned int sidx,
                             unsigned int idx, unsigned int t, unsigned int l,
                             int soffset, int offset)
{
#pragma unroll
    for (unsigned int d=1; d<=STENCIL_DEPTH; d++) *u += (t+d < l) ? smem[sidx+d*soffset] : d_u1[idx+d*offset];
}

__device__ void smem_stencil(float* smem, float* d_u1, float* d_u2, unsigned int sidx, unsigned int idx)
{
    float u = 0.0f;
    accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.x, 1, 1);
    accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.x, BLOCK_X, 1, 1);
#if DIMENSIONS>1
    accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.y, SMEM_X, NX);
    accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.y, BLOCK_Y, SMEM_X, NX);
#endif
#if DIMENSIONS>2
    accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.z, SMEM_X*BLOCK_Y, NX*NY);
    accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.z, BLOCK_Z, SMEM_X*BLOCK_Y, NX*NY);
#endif
    d_u2[idx] = u / STENCIL_COEFF - smem[sidx];
}

__device__ void smem_unrolled_stencil(
        float *d_u1,
        float *d_u2,
        float *smem,
        unsigned int s,
        unsigned int idx,
        unsigned int sidx)
{
    float u = 0.0f;
    if (s>0)          accumulate_l(&u, smem, sidx, 1);
    else              accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.x, 1, 1);
    if (s+1<UNROLL_X) accumulate_r(&u, smem, sidx, 1);
    else              accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.x, BLOCK_X, 1, 1);
#if DIMENSIONS>1
    accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.y, SMEM_X, NX);
    accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.y, BLOCK_Y, SMEM_X, NX);
#endif
#if DIMENSIONS>2
    accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.z, SMEM_X*BLOCK_Y, NX*NY);
    accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.z, BLOCK_Z, SMEM_X*BLOCK_Y, NX*NY);
#endif
    d_u2[idx] = u / STENCIL_COEFF - smem[sidx];
}

__device__ void smem_reg_stencil(float* smem, float* d_u2, float* yval, unsigned int sidx, unsigned int idx)
{
    float u = 0.0f;
    accumulate_l(&u, smem, sidx, 1);
    accumulate_r(&u, smem, sidx, 1);
#if DIMENSIONS>1
    accumulate_reg_l(&u, yval);
    accumulate_reg_r(&u, yval);
#endif
    //return u;
    d_u2[idx] = u / STENCIL_COEFF - yval[STENCIL_DEPTH];
}

__device__ void smem_padded_stencil(
        float *smem,
        float *d_u2,
        unsigned int idx,
        unsigned int sidx)
{
    float u = 0.0f;
    accumulate_l(&u, smem, sidx, 1);
    accumulate_r(&u, smem, sidx, 1);
#if DIMENSIONS>1
    accumulate_l(&u, smem, sidx, SMEM_P_X);
    accumulate_r(&u, smem, sidx, SMEM_P_X);
#endif
#if DIMENSIONS>2
    accumulate_l(&u, smem, sidx, SMEM_P_X*SMEM_P_Y);
    accumulate_r(&u, smem, sidx, SMEM_P_X*SMEM_P_Y);
#endif
    d_u2[idx] = u / STENCIL_COEFF - smem[sidx];
}

#endif // STENCILS_CU
