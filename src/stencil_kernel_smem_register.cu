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
    {
        prefetch_i_left(i, sidx, idx, smem, d_u1);
        prefetch_i_right(i, sidx, idx, smem, d_u1);
        prefetch_reg_j_down(j, idx, yval, d_u1);
        prefetch_reg_j_up(j, idx, yval, d_u1);
        yval[STENCIL_DEPTH] = d_u1[idx];
        smem[sidx] = yval[STENCIL_DEPTH];
    }
    this_thread_block().sync();
    if (check_stencil_border_2d(i, j, jstart, jend))
    {
        d_u2[idx] = smem_reg_stencil(smem, yval, sidx) / STENCIL_COEFF - yval[STENCIL_DEPTH];
    }
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
        {
            if (s==0)          prefetch_i_left(i_off, sidx, idx, smem, d_u1);
            if (s==UNROLL_X-1) prefetch_i_right(i_off, sidx, idx, smem, d_u1);
            prefetch_reg_j_down(j, idx, ys, d_u1);
            prefetch_reg_j_up(j, idx, ys, d_u1);
            ys[STENCIL_DEPTH] = d_u1[idx];
            smem[sidx] = ys[STENCIL_DEPTH];
        }
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
        {
            d_u2[idx] = smem_reg_stencil(smem, ys, sidx) / STENCIL_COEFF - ys[STENCIL_DEPTH];
        }
    }
}

