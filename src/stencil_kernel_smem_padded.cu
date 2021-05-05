#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "prefetch_smem.cu"
#include "stencils_border_check.cu"
using namespace cooperative_groups;

__global__ void smem_padded_2d(float* __restrict__ d_u1,
                                 float* __restrict__ d_u2,
                                 unsigned int jstart,
                                 unsigned int jend)
{
    extern __shared__ float smem[];
    unsigned int i, j, idx, sidx;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j*NX;
    sidx = (threadIdx.x + STENCIL_DEPTH) + (threadIdx.y + STENCIL_DEPTH)*SMEM_P_X;
    prefetch(smem, d_u1, 0, i, j, idx, sidx, jstart, jend);
    this_thread_block().sync();
    smem_padded_stencil(smem, d_u2, i, j, idx, sidx, jstart, jend);
}

__global__ void smem_padded_unroll_2d(float* __restrict__ d_u1,
                                        float* __restrict__ d_u2,
                                        unsigned int jstart,
                                        unsigned int jend)
{
    extern __shared__ float smem[];
    unsigned int i, j, s, idx, sidx, ioff, si, sj;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    si = threadIdx.x + STENCIL_DEPTH;
    sj = threadIdx.y + STENCIL_DEPTH;
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = si+ioff + sj*SMEM_P_X;
        prefetch(smem, d_u1, s, i+ioff, j, idx, sidx, jstart, jend);
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = (si+ioff) + sj*SMEM_P_X;
        smem_padded_stencil(smem, d_u2, i+ioff, j, idx, sidx, jstart, jend);
    }
}
