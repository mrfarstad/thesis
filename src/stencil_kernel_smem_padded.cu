#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "prefetch_smem.cu"
#include "stencil_border_check.cu"
using namespace cooperative_groups;

__global__ void smem_padded_3d(float* __restrict__ d_u1,
                               float* __restrict__ d_u2,
                               unsigned int kstart,
                               unsigned int kend)
{

    extern __shared__ float smem[];
    unsigned int i, j, k, idx, sidx, smem_p_x, smem_p_y;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    k  = threadIdx.z + blockIdx.z*blockDim.z;
    idx = i + j*NX + k*NX*NY;
    smem_p_x = blockDim.x*COARSEN_X+2*RADIUS;
    smem_p_y = blockDim.y+2*RADIUS;
    sidx = (threadIdx.x + RADIUS)
         + (threadIdx.y + RADIUS)*smem_p_x
         + (threadIdx.z + RADIUS)*smem_p_x*smem_p_y;
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
    unsigned int i, j, k, si, sj, sk, i_off, si_off, u, idx, sidx, smem_p_x, smem_p_y;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    k  = threadIdx.z + blockIdx.z*blockDim.z;
    si = threadIdx.x + RADIUS;
    sj = threadIdx.y + RADIUS;
    sk = threadIdx.z + RADIUS;
    smem_p_x = blockDim.x*COARSEN_X+2*RADIUS;
    smem_p_y = blockDim.y+2*RADIUS;
#pragma unroll
    for (u=0; u<COARSEN_X; u++) {
        i_off  = i+u*blockDim.x;
        si_off = si+u*blockDim.x;
        idx    = i_off+j*NX+k*NX*NY;
        sidx = si_off + sj*smem_p_x + sk*smem_p_x*smem_p_y;
        if (check_domain_border_3d(i_off, j, k, kstart, kend))
            prefetch_3d(smem, d_u1, u, i_off, j, k, idx, sidx, kstart, kend);
    }
    this_thread_block().sync();
#pragma unroll
    for (u=0; u<COARSEN_X; u++) {
        i_off  = i+u*blockDim.x;
        idx    = i_off+j*NX+k*NX*NY;
        si_off = si+u*blockDim.x;
        sidx   = si_off + sj*smem_p_x + sk*smem_p_x*smem_p_y;
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
    unsigned int i, j, idx, sidx, smem_p_x;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    idx = i + j*NX;
    smem_p_x = blockDim.x*COARSEN_X+2*RADIUS;
    sidx = (threadIdx.x + RADIUS)
         + (threadIdx.y + RADIUS)*smem_p_x;
    if (check_domain_border_2d(i, j, jstart, jend))
        prefetch_2d(smem, d_u1, 0, i, j, idx, sidx, jstart, jend);
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
    unsigned int i, j, si, sj, i_off, si_off, u, idx, sidx, smem_p_x;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    si = threadIdx.x + RADIUS;
    sj = threadIdx.y + RADIUS;
    smem_p_x = blockDim.x*COARSEN_X+2*RADIUS;
#pragma unroll
    for (u=0; u<COARSEN_X; u++) {
        i_off  = i+u*blockDim.x;
        si_off = si+u*blockDim.x;
        idx    = i_off+j*NX;
        sidx   = si_off+sj*smem_p_x;
        if (check_domain_border_2d(i_off, j, jstart, jend))
            prefetch_2d(smem, d_u1, u, i_off, j, idx, sidx, jstart, jend);
    }
    this_thread_block().sync();
#pragma unroll
    for (u=0; u<COARSEN_X; u++) {
        i_off  = i+u*blockDim.x;
        idx    = i_off+j*NX;
        si_off = si+u*blockDim.x;
        sidx   = si_off+sj*smem_p_x;
        if (check_stencil_border_2d(i_off, j, jstart, jend))
            smem_padded_stencil(smem, d_u2, idx, sidx);
    }
}
