#ifndef PREFETCH_SMEM_CU
#define PREFETCH_SMEM_CU

#include "../include/constants.h"
#include "stencils_border_check.cu"

__device__ void prefetch_reg_j_down(
    unsigned int j,
    unsigned int idx,
    float *yval,
    float *d_u1)
{
    if (j >= STENCIL_DEPTH)
        for (unsigned int s = 0; s < STENCIL_DEPTH; s++)
            yval[s] = d_u1[idx + (s - STENCIL_DEPTH) * NX];
}

__device__ void prefetch_reg_j_up(
    unsigned int j,
    unsigned int idx,
    float *yval,
    float *d_u1)
{
    if (j < NY-STENCIL_DEPTH)
        for (unsigned int s = STENCIL_DEPTH+1; s < 2*STENCIL_DEPTH+1; s++)
            yval[s] = d_u1[idx + (s - STENCIL_DEPTH) * NX];
}

__device__ void prefetch_i_left(
    unsigned int i,
    unsigned int sidx,
    unsigned int idx,
    float *smem,
    float *d_u1)
{
    if (threadIdx.x < STENCIL_DEPTH && i >= STENCIL_DEPTH)
    {
        smem[sidx-STENCIL_DEPTH] = d_u1[idx-STENCIL_DEPTH];
    }
}

__device__ void prefetch_i_right(
    unsigned int i,
    unsigned int sidx,
    unsigned int idx,
    float *smem,
    float *d_u1)
{
    if (threadIdx.x >= BLOCK_X-STENCIL_DEPTH && i < NX-STENCIL_DEPTH)
    {
        smem[sidx+STENCIL_DEPTH] = d_u1[idx+STENCIL_DEPTH];
    }
}

__device__ void prefetch(
    float *smem,
    float *d_u1,
    unsigned int s,
    unsigned int i,
    unsigned int j,
    unsigned int idx,
    unsigned int sidx,
    unsigned int jstart,
    unsigned int jend)
{
    if(s==0)          prefetch_i_left(i, sidx, idx, smem, d_u1);
    if(s==UNROLL_X-1) prefetch_i_right(i, sidx, idx, smem, d_u1);
    if (threadIdx.y < STENCIL_DEPTH && j >= jstart+STENCIL_DEPTH)
    {
        smem[sidx-STENCIL_DEPTH*SMEM_P_X] = d_u1[idx-STENCIL_DEPTH*NX];
    }
    if (threadIdx.y >= BLOCK_Y-STENCIL_DEPTH && j <= jend-STENCIL_DEPTH)
    {
        smem[sidx+STENCIL_DEPTH*SMEM_P_X] = d_u1[idx+STENCIL_DEPTH*NX];
    }
    smem[sidx] = d_u1[idx];
}
#endif // PREFETCH_SMEM_CU
