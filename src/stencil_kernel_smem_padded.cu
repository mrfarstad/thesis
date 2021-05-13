#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "prefetch_smem.cu"
#include "stencils_border_check.cu"
using namespace cooperative_groups;

__global__ void smem_padded_3d(float* __restrict__ d_u1,
                               float* __restrict__ d_u2,
                               unsigned int kstart,
                               unsigned int kend)
{

    extern __shared__ float smem[];
    unsigned int i, j, k, idx, sidx;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    idx = i + j*NX + k*NX*NY;
    sidx = (threadIdx.x + STENCIL_DEPTH)
         + (threadIdx.y + STENCIL_DEPTH)*SMEM_P_X
         + (threadIdx.z + STENCIL_DEPTH)*SMEM_P_X*SMEM_P_Y;
    if (check_domain_border_3d(i, j, k, kstart, kend))
        prefetch_3d(smem, d_u1, 0, i, j, k, idx, sidx, kstart, kend);
    this_thread_block().sync();
    if (check_stencil_border_3d(i, j, k, kstart, kend))
        smem_padded_stencil(smem, d_u2, idx, sidx);
}

__global__ void smem_padded_unroll_3d(float* __restrict__ d_u1,
                                      float* __restrict__ d_u2,
                                      unsigned int kstart,
                                      unsigned int kend)
{
    extern __shared__ float smem[];
    unsigned int i, j, k, si, sj, sk, i_off, si_off, u, idx, sidx;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    si = threadIdx.x + STENCIL_DEPTH;
    sj = threadIdx.y + STENCIL_DEPTH;
    sk = threadIdx.z + STENCIL_DEPTH;
#pragma unroll
    for (u=0; u<UNROLL_X; u++) {
        i_off  = i+u*BLOCK_X;
        si_off = si+u*BLOCK_X;
        idx    = i_off+j*NX+k*NX*NY;
        sidx = si_off + sj*SMEM_P_X + sk*SMEM_P_X*SMEM_P_Y;
        if (check_domain_border_3d(i_off, j, k, kstart, kend))
            prefetch_3d(smem, d_u1, u, i_off, j, k, idx, sidx, kstart, kend);
    }
    this_thread_block().sync();
#pragma unroll
    for (u=0; u<UNROLL_X; u++) {
        i_off  = i+u*BLOCK_X;
        idx    = i_off+j*NX+k*NX*NY;
        si_off = si+u*BLOCK_X;
        sidx   = si_off + sj*SMEM_P_X + sk*SMEM_P_X*SMEM_P_Y;
        if (check_stencil_border_3d(i_off, j, k, kstart, kend))
            smem_padded_stencil(smem, d_u2, idx, sidx);
    }
}

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
    if (check_domain_border_2d(i, j, jstart, jend))
        prefetch(smem, d_u1, 0, i, j, idx, sidx, jstart, jend);
    this_thread_block().sync();
    if (check_stencil_border_2d(i, j, jstart, jend))
        smem_padded_stencil(smem, d_u2, idx, sidx);
}

__global__ void smem_padded_unroll_2d(float* __restrict__ d_u1,
                                      float* __restrict__ d_u2,
                                      unsigned int jstart,
                                      unsigned int jend)
{
    extern __shared__ float smem[];
    unsigned int i, j, si, sj, i_off, si_off, u, idx, sidx;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    si = threadIdx.x + STENCIL_DEPTH;
    sj = threadIdx.y + STENCIL_DEPTH;
#pragma unroll
    for (u=0; u<UNROLL_X; u++) {
        i_off  = i+u*BLOCK_X;
        si_off = si+u*BLOCK_X;
        idx    = i_off+j*NX;
        sidx   = si_off+sj*SMEM_P_X;
        if (check_domain_border_2d(i_off, j, jstart, jend))
            prefetch(smem, d_u1, u, i_off, j, idx, sidx, jstart, jend);
    }
    this_thread_block().sync();
#pragma unroll
    for (u=0; u<UNROLL_X; u++) {
        i_off  = i+u*BLOCK_X;
        idx    = i_off+j*NX;
        si_off = si+u*BLOCK_X;
        sidx   = si_off+sj*SMEM_P_X;
        if (check_stencil_border_2d(i_off, j, jstart, jend))
            smem_padded_stencil(smem, d_u2, idx, sidx);
    }
}
