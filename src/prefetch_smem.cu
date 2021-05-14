#ifndef PREFETCH_SMEM_CU
#define PREFETCH_SMEM_CU

#include "../include/constants.h"
#include "stencils_border_check.cu"

__device__ void prefetch_l(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int t,
    unsigned int didx,
    unsigned int didx_lim,
    unsigned int soffset,
    unsigned int offset)
{
    if (t < STENCIL_DEPTH && didx >= didx_lim)
    {
        smem[sidx-STENCIL_DEPTH*soffset] = d_u1[idx-STENCIL_DEPTH*offset];
    }
}

__device__ void prefetch_r(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int t,
    unsigned int t_lim,
    unsigned int didx,
    unsigned int didx_lim,
    unsigned int soffset,
    unsigned int offset)
{
    if (t >= t_lim && didx <= didx_lim)
    {
        smem[sidx+STENCIL_DEPTH*soffset] = d_u1[idx+STENCIL_DEPTH*offset];
    }
}

__device__ void prefetch_i_left(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int i)
{
    prefetch_l(smem, d_u1, sidx, idx, threadIdx.x, i, STENCIL_DEPTH, 1, 1);
}

__device__ void prefetch_i_right(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int i)
{
    prefetch_r(smem, d_u1, sidx, idx, threadIdx.x, BLOCK_X-STENCIL_DEPTH, i, NX-STENCIL_DEPTH-1, 1, 1);
}

__device__ void prefetch_j_left(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int j,
    unsigned int jstart)
{
    prefetch_l(smem, d_u1, sidx, idx, threadIdx.y, j, jstart+STENCIL_DEPTH, SMEM_P_X, NX);
}

__device__ void prefetch_j_right(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int j,
    unsigned int jend)
{
    prefetch_r(smem, d_u1, sidx, idx, threadIdx.y, BLOCK_Y-STENCIL_DEPTH, j, jend-STENCIL_DEPTH, SMEM_P_X, NX);
}

__device__ void prefetch_k_left(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int k,
    unsigned int kstart)
{
    prefetch_l(smem, d_u1, sidx, idx, threadIdx.z, k, kstart+STENCIL_DEPTH, SMEM_P_X*SMEM_P_Y, NX*NY);
}

__device__ void prefetch_k_right(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int k,
    unsigned int kend)
{
    prefetch_r(smem, d_u1, sidx, idx, threadIdx.z, BLOCK_Z-STENCIL_DEPTH, k, kend-STENCIL_DEPTH, SMEM_P_X*SMEM_P_Y, NX*NY);
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
    if(s==0)          prefetch_i_left(smem, d_u1, sidx, idx, i);
    if(s==UNROLL_X-1) prefetch_i_right(smem, d_u1, sidx, idx, i);
    prefetch_j_left(smem, d_u1, sidx, idx, j, jstart);
    prefetch_j_right(smem, d_u1, sidx, idx, j, jend);
    smem[sidx] = d_u1[idx];
}

__device__ void prefetch_3d(
    float *smem,
    float *d_u1,
    unsigned int s,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int idx,
    unsigned int sidx,
    unsigned int kstart,
    unsigned int kend)
{
    if(s==0)          prefetch_i_left(smem, d_u1, sidx, idx, i);
    if(s==UNROLL_X-1) prefetch_i_right(smem, d_u1, sidx, idx, i);
    prefetch_j_left(smem, d_u1, sidx, idx, j, 0);
    prefetch_j_right(smem, d_u1, sidx, idx, j, NY-1);
    prefetch_k_left(smem, d_u1, sidx, idx, k, kstart);
    prefetch_k_right(smem, d_u1, sidx, idx, k, kend);
    smem[sidx] = d_u1[idx];
}

__device__ void prefetch_reg_l(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int didx,
    unsigned int didx_lim,
    unsigned int offset)
{
    if (didx >= didx_lim)
        for (unsigned int s = 0; s < STENCIL_DEPTH; s++)
            yval[s] = d_u1[idx + (s - STENCIL_DEPTH) * offset];
}

__device__ void prefetch_reg_r(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int didx,
    unsigned int didx_lim,
    unsigned int offset)
{
    if (didx <= didx_lim)
        for (unsigned int s = STENCIL_DEPTH+1; s < 2*STENCIL_DEPTH+1; s++)
            yval[s] = d_u1[idx + (s - STENCIL_DEPTH) * offset];
}

__device__ void prefetch_reg_j_down(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int j,
    unsigned int jstart)
{
    prefetch_reg_l(yval, d_u1, idx, j, jstart+STENCIL_DEPTH, NX);
}

__device__ void prefetch_reg_k_down(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int k,
    unsigned int kstart)
{
    prefetch_reg_l(yval, d_u1, idx, k, kstart+STENCIL_DEPTH, NX*NY);
}

__device__ void prefetch_reg_j_up(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int j,
    unsigned int jend)
{
    prefetch_reg_r(yval, d_u1, idx, j, jend-STENCIL_DEPTH, NX);
}

__device__ void prefetch_reg_k_up(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int k,
    unsigned int kend)
{
    prefetch_reg_r(yval, d_u1, idx, k, kend-STENCIL_DEPTH, NX*NY);
}

__device__ void prefetch_register_3d(
    float *smem,
    float *d_u1,
    float *yval,
    unsigned int idx,
    unsigned int sidx,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int start,
    unsigned int end)
{
    prefetch_i_left(smem, d_u1, sidx, idx, i);
    prefetch_i_right(smem, d_u1, sidx, idx, i);
    prefetch_j_left(smem, d_u1, sidx, idx, j, 0);
    prefetch_j_right(smem, d_u1, sidx, idx, j, NY-1);
    prefetch_reg_k_down(yval, d_u1, idx, k, start);
    prefetch_reg_k_up(yval, d_u1, idx, k, end);
    yval[STENCIL_DEPTH] = d_u1[idx];
    smem[sidx] = yval[STENCIL_DEPTH];
}

__device__ void prefetch_register(
    float *smem,
    float *d_u1,
    float *yval,
    unsigned int idx,
    unsigned int sidx,
    unsigned int i,
    unsigned int j,
    unsigned int start,
    unsigned int end)
{
    prefetch_i_left(smem, d_u1, sidx, idx, i);
    prefetch_i_right(smem, d_u1, sidx, idx, i);
    prefetch_reg_j_down(yval, d_u1, idx, j, start);
    prefetch_reg_j_up(yval, d_u1, idx, j, end);
    yval[STENCIL_DEPTH] = d_u1[idx];
    smem[sidx] = yval[STENCIL_DEPTH];
}

__device__ void prefetch_register_unroll_3d(
    float *smem,
    float *d_u1,
    float *yval,
    unsigned int s,
    unsigned int idx,
    unsigned int sidx,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int kstart,
    unsigned int kend)
{
    if (s==0)          prefetch_i_left(smem, d_u1, sidx, idx, i);
    if (s==UNROLL_X-1) prefetch_i_right(smem, d_u1, sidx, idx, i);
    prefetch_j_left(smem, d_u1, sidx, idx, j, 0);
    prefetch_j_right(smem, d_u1, sidx, idx, j, NY-1);
    prefetch_reg_k_down(yval, d_u1, idx, k, kstart);
    prefetch_reg_k_up(yval, d_u1, idx, k, kend);
    yval[STENCIL_DEPTH] = d_u1[idx];
    smem[sidx] = yval[STENCIL_DEPTH];
}

__device__ void prefetch_register_unroll(
    float *smem,
    float *d_u1,
    float *yval,
    unsigned int s,
    unsigned int idx,
    unsigned int sidx,
    unsigned int i,
    unsigned int j,
    unsigned int jstart,
    unsigned int jend)
{
    if (s==0)          prefetch_i_left(smem, d_u1, sidx, idx, i);
    if (s==UNROLL_X-1) prefetch_i_right(smem, d_u1, sidx, idx, i);
    prefetch_reg_j_down(yval, d_u1, idx, j, jstart);
    prefetch_reg_j_up(yval, d_u1, idx, j, jend);
    yval[STENCIL_DEPTH] = d_u1[idx];
    smem[sidx] = yval[STENCIL_DEPTH];
}

#endif // PREFETCH_SMEM_CU
