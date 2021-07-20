#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "prefetch_smem.cu"
#include "stencil_border_check.cu"
using namespace cooperative_groups;

__global__ void smem_register_3d(float* __restrict__ d_u1,
                                 float* __restrict__ d_u2,
                                 unsigned int kstart,
                                 unsigned int kend)
{
    unsigned int i, j, k, idx, sidx;
    extern __shared__ float smem[];
    float yval[REG_SIZE];
    i  = threadIdx.x + blockIdx.x*blockDim.x;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    k  = threadIdx.z + blockIdx.z*blockDim.z;
    idx = i + j*NX + k*NX*NY;
    sidx = (threadIdx.x + RADIUS)
         + (threadIdx.y + RADIUS)*SMEM_P_X
         + (threadIdx.z)*SMEM_P_X*SMEM_P_Y;
    if (check_domain_border_3d(i, j, k, kstart, kend))
        prefetch_register_3d(smem, d_u1, yval, 0, idx, sidx, i, j, k, kstart, kend);
    this_thread_block().sync();
    if (check_stencil_border_3d(i, j, k, kstart, kend))
        smem_register_stencil(smem, d_u2, yval, sidx, idx);
}

__global__ void smem_register_unroll_3d(float* __restrict__ d_u1,
                                        float* __restrict__ d_u2,
                                        unsigned int kstart,
                                        unsigned int kend)
{
    unsigned int i, j, k, s, si, sj, i_off, si_off, idx, sidx;
    extern __shared__ float smem[];
    float yval[COARSEN_X][REG_SIZE];
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    k  = threadIdx.z + blockIdx.z*blockDim.z;
    si = threadIdx.x + RADIUS;
    sj = threadIdx.y + RADIUS;
#pragma unroll
    for (s=0; s<COARSEN_X; s++) {
        i_off = i + s*blockDim.x;
        idx = i_off + j*NX + k*NX*NY;
        si_off = si + s*blockDim.x;
        sidx = si_off + sj*SMEM_P_X + threadIdx.z*SMEM_P_X*SMEM_P_Y;
        if (check_domain_border_3d(i_off, j, k, kstart, kend))
            prefetch_register_3d(smem, d_u1, yval[s], s, idx, sidx, i_off, j, k, kstart, kend);
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<COARSEN_X; s++) {
        i_off = i + s*blockDim.x;
        idx = i_off + j*NX + k*NX*NY;
        si_off = si + s*blockDim.x;
        sidx = si_off + sj*SMEM_P_X + threadIdx.z*SMEM_P_X*SMEM_P_Y;
        if (check_stencil_border_3d(i_off, j, k, kstart, kend))
            smem_register_stencil(smem, d_u2, yval[s], sidx, idx);
    }
}

__global__ void smem_register_2d(float* __restrict__ d_u1,
                                 float* __restrict__ d_u2,
                                 unsigned int jstart,
                                 unsigned int jend)
{
    unsigned int i, j, idx, sidx;
    extern __shared__ float smem[];
    float yval[REG_SIZE];
    i  = threadIdx.x + blockIdx.x*blockDim.x;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    idx = i + j*NX;
    sidx = (threadIdx.x + RADIUS) + threadIdx.y*SMEM_P_X;
    if (check_domain_border_2d(i, j, jstart, jend))
        prefetch_register(smem, d_u1, yval, 0, idx, sidx, i, j, jstart, jend);
    this_thread_block().sync();
    if (check_stencil_border_2d(i, j, jstart, jend))
        smem_register_stencil(smem, d_u2, yval, sidx, idx);
}

__global__ void smem_register_unroll_2d(float* __restrict__ d_u1,
                                        float* __restrict__ d_u2,
                                        unsigned int jstart,
                                        unsigned int jend)
{
    unsigned int i, j, s, si, i_off, idx, sidx, ioff;
    extern __shared__ float smem[];
    float yval[COARSEN_X][REG_SIZE], *ys;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    idx = i + j*NX;
    si = threadIdx.x + RADIUS;
#pragma unroll
    for (s=0; s<COARSEN_X; s++) {
        ioff = s*blockDim.x;
        i_off = i + ioff;
        idx = i_off + j*NX;
        sidx = (si+ioff) + threadIdx.y*SMEM_P_X;
        ys = yval[s];
        if (check_domain_border_2d(i_off, j, jstart, jend))
            prefetch_register(smem, d_u1, ys, s, idx, sidx, i_off, j, jstart, jend);
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<COARSEN_X; s++) {
        ioff = s*blockDim.x;
        i_off = i + ioff;
        idx = i_off + j*NX;
        sidx = (si+ioff) + threadIdx.y*SMEM_P_X;
        ys = yval[s];
        if (check_stencil_border_2d(i_off, j, jstart, jend))
            smem_register_stencil(smem, d_u2, ys, sidx, idx);
    }
}

