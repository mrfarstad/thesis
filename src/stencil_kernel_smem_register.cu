#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "prefetch_smem.cu"
#include "stencils_border_check.cu"
using namespace cooperative_groups;

__global__ void smem_register_2d(float* __restrict__ d_u1,
                                 float* __restrict__ d_u2,
                                 unsigned int jstart,
                                 unsigned int jend)
{
    unsigned int i, j, idx, sidx;
    extern __shared__ float smem[];
    float yval[2*STENCIL_DEPTH+1];
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j*NX;
    sidx = (threadIdx.x + STENCIL_DEPTH) + threadIdx.y*SMEM_P_X;
    if (check_domain_border_2d(i, j, jstart, jend))
        prefetch_register(smem, d_u1, yval, idx, sidx, i, j, jstart, jend);
    this_thread_block().sync();
    if (check_stencil_border_2d(i, j, jstart, jend))
        smem_reg_stencil(smem, d_u2, yval, sidx, idx);
}

__global__ void smem_register_unroll_2d(float* __restrict__ d_u1,
                                        float* __restrict__ d_u2,
                                        unsigned int jstart,
                                        unsigned int jend)
{
    unsigned int i, j, s, si, i_off, idx, sidx, ioff;
    extern __shared__ float smem[];
    float yval[UNROLL_X][REG_SIZE], *ys;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j*NX;
    si = threadIdx.x + STENCIL_DEPTH;
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        i_off = i + ioff;
        idx = i_off + j*NX;
        sidx = (si+ioff) + threadIdx.y*SMEM_P_X;
        ys = yval[s];
        if (check_domain_border_2d(i_off, j, jstart, jend))
            prefetch_register_unroll(smem, d_u1, ys, s, idx, sidx, i_off, j, jstart, jend);
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        i_off = i + ioff;
        idx = i_off + j*NX;
        sidx = (si+ioff) + threadIdx.y*SMEM_P_X;
        ys = yval[s];
        if (check_stencil_border_2d(i_off, j, jstart, jend))
            smem_reg_stencil(smem, d_u2, ys, sidx, idx);
    }
}

