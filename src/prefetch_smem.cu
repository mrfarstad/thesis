#ifndef PREFETCH_SMEM_CU
#define PREFETCH_SMEM_CU

#include "../include/constants.h"
#include "stencil_border_check.cu"

__device__ void prefetch_prev(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int thread_id,
    unsigned int domain_idx,
    unsigned int domain_idx_limit,
    unsigned int soffset,
    unsigned int offset)
{
    if (thread_id < STENCIL_DEPTH && domain_idx >= domain_idx_limit)
    {
        smem[sidx-STENCIL_DEPTH*soffset] = d_u1[idx-STENCIL_DEPTH*offset];
    }
}

__device__ void prefetch_next(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int thread_id,
    unsigned int thread_id_limit,
    unsigned int domain_idx,
    unsigned int domain_idx_limit,
    unsigned int soffset,
    unsigned int offset)
{
    if (thread_id >= thread_id_limit && domain_idx <= domain_idx_limit)
    {
        smem[sidx+STENCIL_DEPTH*soffset] = d_u1[idx+STENCIL_DEPTH*offset];
    }
}

__device__ void prefetch_i_prev(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int i)
{
    prefetch_prev(smem, d_u1, sidx, idx, threadIdx.x, i, STENCIL_DEPTH, 1, 1);
}

__device__ void prefetch_i_next(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int i)
{
    prefetch_next(smem, d_u1, sidx, idx, threadIdx.x, blockDim.x-STENCIL_DEPTH, i, NX-STENCIL_DEPTH-1, 1, 1);
}

__device__ void prefetch_j_prev(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int j,
    unsigned int jstart)
{
    int smem_p_x = blockDim.x*UNROLL_X+2*STENCIL_DEPTH;
    prefetch_prev(smem, d_u1, sidx, idx, threadIdx.y, j, jstart+STENCIL_DEPTH, smem_p_x, NX);
}

__device__ void prefetch_j_next(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int j,
    unsigned int jend)
{
    int smem_p_x = blockDim.x*UNROLL_X+2*STENCIL_DEPTH;
    prefetch_next(smem, d_u1, sidx, idx, threadIdx.y, blockDim.y-STENCIL_DEPTH, j, jend-STENCIL_DEPTH, smem_p_x, NX);
}

__device__ void prefetch_k_prev(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int k,
    unsigned int kstart)
{
    int smem_p_x = blockDim.x*UNROLL_X+2*STENCIL_DEPTH;
    int smem_p_y = blockDim.y+2*STENCIL_DEPTH;
    prefetch_prev(smem, d_u1, sidx, idx, threadIdx.z, k, kstart+STENCIL_DEPTH, smem_p_x*smem_p_y, NX*NY);
}

__device__ void prefetch_k_next(
    float *smem,
    float *d_u1,
    unsigned int sidx,
    unsigned int idx,
    unsigned int k,
    unsigned int kend)
{
    int smem_p_x = blockDim.x*UNROLL_X+2*STENCIL_DEPTH;
    int smem_p_y = blockDim.y+2*STENCIL_DEPTH;
    prefetch_next(smem, d_u1, sidx, idx, threadIdx.z, blockDim.z-STENCIL_DEPTH, k, kend-STENCIL_DEPTH, smem_p_x*smem_p_y, NX*NY);
}

__device__ void prefetch_2d(
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
    if(s==0)          prefetch_i_prev(smem, d_u1, sidx, idx, i);
    if(s==UNROLL_X-1) prefetch_i_next(smem, d_u1, sidx, idx, i);
    prefetch_j_prev(smem, d_u1, sidx, idx, j, jstart);
    prefetch_j_next(smem, d_u1, sidx, idx, j, jend);
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
    if(s==0)          prefetch_i_prev(smem, d_u1, sidx, idx, i);
    if(s==UNROLL_X-1) prefetch_i_next(smem, d_u1, sidx, idx, i);
    prefetch_j_prev(smem, d_u1, sidx, idx, j, 0);
    prefetch_j_next(smem, d_u1, sidx, idx, j, NY-1);
    prefetch_k_prev(smem, d_u1, sidx, idx, k, kstart);
    prefetch_k_next(smem, d_u1, sidx, idx, k, kend);
    smem[sidx] = d_u1[idx];
}

__device__ void prefetch_register_prev(
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

__device__ void prefetch_register_next(
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

__device__ void prefetch_register_j_prev(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int j,
    unsigned int jstart)
{
    prefetch_register_prev(yval, d_u1, idx, j, jstart+STENCIL_DEPTH, NX);
}

__device__ void prefetch_register_k_prev(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int k,
    unsigned int kstart)
{
    prefetch_register_prev(yval, d_u1, idx, k, kstart+STENCIL_DEPTH, NX*NY);
}

__device__ void prefetch_register_j_next(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int j,
    unsigned int jend)
{
    prefetch_register_next(yval, d_u1, idx, j, jend-STENCIL_DEPTH, NX);
}

__device__ void prefetch_register_k_next(
    float *yval,
    float *d_u1,
    unsigned int idx,
    unsigned int k,
    unsigned int kend)
{
    prefetch_register_next(yval, d_u1, idx, k, kend-STENCIL_DEPTH, NX*NY);
}

__device__ void prefetch_register_3d(
    float *smem,
    float *d_u1,
    float *yval,
    unsigned int s,
    unsigned int idx,
    unsigned int sidx,
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int start,
    unsigned int end)
{
    if (s==0)          prefetch_i_prev(smem, d_u1, sidx, idx, i);
    if (s==UNROLL_X-1) prefetch_i_next(smem, d_u1, sidx, idx, i);
    prefetch_j_prev(smem, d_u1, sidx, idx, j, 0);
    prefetch_j_next(smem, d_u1, sidx, idx, j, NY-1);
    prefetch_register_k_prev(yval, d_u1, idx, k, start);
    prefetch_register_k_next(yval, d_u1, idx, k, end);
    yval[STENCIL_DEPTH] = d_u1[idx];
    smem[sidx] = yval[STENCIL_DEPTH];
}

__device__ void prefetch_register(
    float *smem,
    float *d_u1,
    float *yval,
    unsigned int s,
    unsigned int idx,
    unsigned int sidx,
    unsigned int i,
    unsigned int j,
    unsigned int start,
    unsigned int end)
{
    if (s==0)          prefetch_i_prev(smem, d_u1, sidx, idx, i);
    if (s==UNROLL_X-1) prefetch_i_next(smem, d_u1, sidx, idx, i);
    prefetch_register_j_prev(yval, d_u1, idx, j, start);
    prefetch_register_j_next(yval, d_u1, idx, j, end);
    yval[STENCIL_DEPTH] = d_u1[idx];
    smem[sidx] = yval[STENCIL_DEPTH];
}

#endif // PREFETCH_SMEM_CU
