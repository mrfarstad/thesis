#include "../include/constants.h"

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
    unsigned int s,
    unsigned int i,
    unsigned int j,
    unsigned int idx,
    unsigned int sidx,
    float *smem,
    float *d_u1)
{
    if (i<NX && j<NY)
    {
        if(s==0)          prefetch_i_left(i, sidx, idx, smem, d_u1);
        if(s==UNROLL_X-1) prefetch_i_right(i, sidx, idx, smem, d_u1);
        if (threadIdx.y < STENCIL_DEPTH && j >= STENCIL_DEPTH)
        {
            smem[sidx-STENCIL_DEPTH*SMEM_P_X] = d_u1[idx-STENCIL_DEPTH*NX];
        }
        if (threadIdx.y >= BLOCK_Y-STENCIL_DEPTH && j < NY-STENCIL_DEPTH)
        {
            smem[sidx+STENCIL_DEPTH*SMEM_P_X] = d_u1[idx+STENCIL_DEPTH*NX];
        }
        smem[sidx] = d_u1[idx];
    }
}

